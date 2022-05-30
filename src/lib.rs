// TODO: General crate docs + deny(missing_docs)

use nom::AsBytes;
use std::rc::Rc;

/// We accept as input types which can be infaillibly converted into a sequence
/// of bytes and faillibly converted from some sequences of bytes.
pub trait ByteBased: AsBytes {
    /// Try constructing the type from bytes
    fn try_from(bytes: &[u8]) -> Option<&Self>;

    /// Length of the byte representation of this type
    fn len(&self) -> usize {
        self.as_bytes().len()
    }

    /// Slice the first N bytes of the input, and return the result if valid
    fn get_prefix(&self, len: usize) -> Option<&Self> {
        self.as_bytes().get(..len).map(Self::try_from).flatten()
    }
}
//
impl ByteBased for [u8] {
    fn try_from(bytes: &[u8]) -> Option<&[u8]> {
        Some(bytes)
    }
}
//
impl ByteBased for str {
    fn try_from(bytes: &[u8]) -> Option<&str> {
        std::str::from_utf8(bytes).ok()
    }

    fn get_prefix(&self, len: usize) -> Option<&Self> {
        self.get(..len)
    }
}

/// Cache of strings we've parsed before and associated parser output
pub struct CachingParser<Input: ByteBased + ?Sized, Output> {
    /// Cached (string -> output) mappings
    data: PrefixToOutput<Output>,

    /// Cache configuration
    config: CachingParserConfig<Input, Output>,
}
//
/// Mapping from a chunk of parser input (prefix string) to associated output or
/// possibly next strings (suffixes) eventually leading to parser output.
type PrefixToOutput<Output> = Vec<(Vec<u8>, PrefixMapping<Output>)>;
//
/// Things which a prefix string can map into
#[derive(Clone, Debug, PartialEq)]
enum PrefixMapping<Output> {
    /// Direct mapping to output data
    Output(Rc<Output>),

    /// Indirect mapping via suffixes
    Suffixes(PrefixToOutput<Output>),
}
//
struct CachingParserConfig<Input: ByteBased + ?Sized, Output> {
    /// Inner parser that we're trying to avoid calling via memoization
    ///
    /// See constructor docs for semantics.
    ///
    // TODO: Generalize to any nom parser
    parser: Box<dyn Fn(&Input) -> Option<(&Input, Output)>>,

    /// Check that a parser output estimated from the cache is valid
    ///
    /// See constructor docs for semantics.
    ///
    check_result: Box<dyn Fn(&Input, &Output) -> bool>,

    /// Truth that a parse was complex enough to warrant memoization
    ///
    /// Receives as input the input that was parsed and the output that was
    /// produced, returns as output the truth that this is worth caching.
    ///
    worth_caching: Box<dyn Fn(&Input, &Output) -> bool>,

    /// Size of a PrefixToOutput list that initiates prefix deduplication
    high_water_mark: usize,

    /// Targeted max PrefixToOutput list length after prefix deduplication,
    /// as a fraction of high_water_mark, guides deduplication recursion.
    low_water_mark_ratio: f32,
}
//
impl<Input: ByteBased + ?Sized, Output> CachingParserConfig<Input, Output> {
    /// Targeted max PrefixToOutput list length after prefix deduplication
    fn low_water_mark(&self) -> usize {
        (self.low_water_mark_ratio * (self.high_water_mark as f32)) as usize
    }
}
//
impl<Input: ByteBased + ?Sized, Output> CachingParser<Input, Output> {
    /// Build a caching parser with default configuration
    ///
    /// `parser` is a nom parser that can be expensive to call, which you are
    /// trying to speed up via memoization.
    ///
    /// `check_result` is a function that can tell if a parser output (input
    /// residual + output) that is estimated from the cache is valid, that is to
    /// say, if it matches what the parser would produce if run on the input
    /// string. This is needed for grammars that end with repetition or optional
    /// terminators, and is typically done by checking for the presence of
    /// terminators in the residual input.
    ///
    pub fn new(
        parser: impl Fn(&Input) -> Option<(&Input, Output)> + 'static,
        check_result: impl Fn(&Input, &Output) -> bool + 'static,
    ) -> Self {
        Self::builder(parser, check_result).build()
    }

    /// Start building a caching parser with non-default settings
    ///
    /// See `CachingParser::new()` for more info on parameter semantics.
    ///
    pub fn builder(
        parser: impl Fn(&Input) -> Option<(&Input, Output)> + 'static,
        check_result: impl Fn(&Input, &Output) -> bool + 'static,
    ) -> CachingParserBuilder<Input, Output> {
        CachingParserBuilder::new(parser, check_result)
    }

    /// Find cached output associated with a string, or compute output and
    /// insert it into the cache.
    // TODO: Just implement the nom Parse trait
    pub fn get_or_insert<'input>(
        &mut self,
        input: &'input Input,
    ) -> Option<(&'input Input, Rc<Output>)> {
        Self::get_or_insert_impl(&mut self.data, &mut self.config, input, input.as_bytes())
    }

    /// Recursive implementation of get_or_insert targeting a cache subtree
    fn get_or_insert_impl<'input>(
        tree: &mut PrefixToOutput<Output>,
        config: &mut CachingParserConfig<Input, Output>,
        initial_input: &'input Input,
        remaining_bytes: &'input [u8],
    ) -> Option<(&'input Input, Rc<Output>)> {
        // Iterate through input prefixes from the current subtree
        for (prefix, mapping) in tree.iter_mut() {
            // If the prefix matches current input...
            if let Some(remaining_bytes) = remaining_bytes.strip_prefix(&**prefix) {
                match mapping {
                    // ...and if we reached the final output...
                    PrefixMapping::Output(o) => {
                        // ...and if the estimated parser output matches
                        // our expectations (UTF-8 remainder) and the user's...
                        if let Some(remaining_input) = Input::try_from(remaining_bytes) {
                            if (config.check_result)(remaining_input, &*o) {
                                // ...then we're done
                                return Some((remaining_input, o.clone()));
                            }
                        }
                    }

                    // If we only ended up on a subtree of suffixes...
                    PrefixMapping::Suffixes(tree) => {
                        // ...then we recurse into that subtree
                        return Self::get_or_insert_impl(
                            tree,
                            config,
                            initial_input,
                            remaining_bytes,
                        );
                    }
                }
            }
        }

        // If no prefix matches, then there is no cached output for this input,
        // and we must run the parser
        let prefix_len = initial_input.len() - remaining_bytes.len();
        let (remaining_input, output) = (config.parser)(initial_input)?;

        // Does this parser result feel worth caching?
        let parsed_len = initial_input.len() - remaining_input.len();
        let parsed_input = initial_input.get_prefix(parsed_len).unwrap();
        let output = if (config.worth_caching)(parsed_input, &output) {
            // If so, memoize it
            let new_suffix = parsed_input.as_bytes()[prefix_len..].into();
            let output = Rc::new(output);
            tree.push((new_suffix, PrefixMapping::Output(output.clone())));

            // Deduplcate the current subtree if high water mark is reached
            if tree.len() >= config.high_water_mark {
                Self::deduplicate(tree, config, 1.0);
            }
            output
        } else {
            Rc::new(output)
        };
        Some((remaining_input, output))
    }

    /// Deduplicate a subtree of the cache
    ///
    /// Identify sets of entries sharing a common prefix, and turn these entries
    /// into subtrees indexed by that common prefix where only the divergent
    /// suffixes remain.
    ///
    /// Recursively deduplicate the subtrees created in this manner until they
    /// are below the low water mark or it is provably impossible for them to go
    /// below said mark (in which case the water mark is adjusted with a warning)
    ///
    fn deduplicate(
        tree: &mut PrefixToOutput<Output>,
        config: &mut CachingParserConfig<Input, Output>,
        water_mark_ratio: f32,
    ) {
        // Extract the old subtree and put a new one in its place
        let mut old_tree = std::mem::take(tree);

        // Prepare to extract entries of the old tree in sorted order
        old_tree.sort_unstable_by(|(k1, _v1), (k2, _v2)| k1.cmp(k2));
        let mut old_tree_iter = old_tree.into_iter().peekable();

        // Pick a reference entry and begin deduplication loop
        let mut reference = old_tree_iter
            .next()
            .expect("tree should contain >= 1 element");
        let mut matches = Vec::new();
        loop {
            // Extract all entries that have 1 byte in common with the reference
            if let Some(&ref_byte) = reference.0.get(0) {
                while let Some(matching) = old_tree_iter.next_if(|(k, _v)| k[0] == ref_byte) {
                    matches.push(matching);
                }
            }

            // If that search yielded at least one result...
            if !matches.is_empty() {
                // ...determine the longest common prefix
                let prefix = Self::lcp(&reference.0[..], matches.iter().map(|(k, _v)| &k[..]));

                // Set up the associated suffix subtree
                let prefix = prefix.to_owned();
                let mut suffixes = std::iter::once(reference)
                    .chain(matches.drain(..))
                    .map(|(mut k, v)| {
                        k.drain(..prefix.len()).for_each(std::mem::drop);
                        (k, v)
                    })
                    .collect::<Vec<_>>();

                // If that subtree is too big, deduplicate it as well
                if suffixes.len() >= config.low_water_mark() {
                    Self::deduplicate(&mut suffixes, config, config.low_water_mark_ratio);
                }

                // Record the deduplicated prefix -> suffixes subtree
                tree.push((prefix, PrefixMapping::Suffixes(suffixes)));
            } else {
                // If no other entry has a prefix in common, keep this one as is
                tree.push(reference);
            }

            // Pick the next reference entry or exit the deduplication loop
            if let Some(new_reference) = old_tree_iter.next() {
                reference = new_reference;
            } else {
                break;
            }
        }

        // Check if the appropriate water mark is now satisfied.
        //
        // If not, the water mark is unsatisfiable, so increase it with a
        // warning to 2x above the max observed irreductible subtree length.
        //
        if tree.len() >= (config.high_water_mark as f32 * water_mark_ratio) as usize {
            config.high_water_mark = (2.0 * tree.len() as f32 / water_mark_ratio) as usize;
            log::warn!(
                "High water mark is unsatisfiable, increasing it to {}",
                config.high_water_mark
            );
        }
    }

    /// Determine the longest common prefix of a set of byte slices, knowing
    /// that their first byte matches
    fn lcp<'a>(reference: &'a [u8], matches: impl Iterator<Item = &'a [u8]> + Clone) -> &'a [u8] {
        // Iterate over reference bytes, skipping the first one which we know
        for (idx, byte) in reference.iter().enumerate().skip(1) {
            // Check that byte for all slices in the deduplication set
            for candidate in matches.clone() {
                match candidate.get(idx) {
                    // If the byte exists and has the same value, continue
                    Some(byte2) if byte2 == byte => {}

                    // Otherwise we found the end of the longest common prefix:
                    // return previous bytes from the reference
                    _ => return &reference[..idx],
                }
            }
        }

        // If all bytes match, the reference is the longest common prefix
        reference
    }
}

/// Mechanism to configure a CachingParser before building it
//
// See method docs for detailed member docs
pub struct CachingParserBuilder<Input: ByteBased + ?Sized, Output> {
    /// Parser to be wrapped
    parser: Box<dyn Fn(&Input) -> Option<(&Input, Output)>>,

    /// Check that a parser output estimated from the cache is valid
    check_result: Box<dyn Fn(&Input, &Output) -> bool>,

    /// Truth that a parse was complex enough to warrant memoization
    worth_caching: Option<Box<dyn Fn(&Input, &Output) -> bool>>,

    /// Size of a PrefixToOutput list that initiates prefix deduplication
    high_water_mark: usize,

    /// Targeted max PrefixToOutput list length after prefix deduplication,
    /// as a fraction of high_water_mark.
    low_water_mark_ratio: f32,
}
//
impl<Input: ByteBased + ?Sized, Output> CachingParserBuilder<Input, Output> {
    /// Start configuring a CachingParser
    ///
    /// See `CachingParser::new()` for more info on parameter semantics.
    ///
    pub fn new(
        parser: impl Fn(&Input) -> Option<(&Input, Output)> + 'static,
        check_result: impl Fn(&Input, &Output) -> bool + 'static,
    ) -> Self {
        Self {
            parser: Box::new(parser),
            check_result: Box::new(check_result),
            worth_caching: None,
            high_water_mark: 512,
            low_water_mark_ratio: 0.5,
        }
    }

    /// Add a criterion for memoizing a parser output or not
    ///
    /// Some parsers are only expensive to call in specific circumstances. In
    /// that case, it is useless to memoize "fast" calls (as cache recalls will
    /// be slower than direct parsing) and doing so can actually harm the memory
    /// footprint and recall performance of the cache. So here you can provide a
    /// criterion to tell which cache outputs should be memoized and which ones
    /// should not be memoized, weeding out excessively simple parses.
    ///
    /// The provided function receives the subset of input that was consumed by
    /// the parser, the output that was produced, and decides on those grounds
    /// whether a certain parse should be memoized (true) or not (false).
    ///
    pub fn retention_criterion(
        mut self,
        worth_caching: impl Fn(&Input, &Output) -> bool + 'static,
    ) -> Self {
        self.worth_caching = Some(Box::new(worth_caching));
        self
    }

    /// Tune the "high water mark" parser cache deduplication parameter
    ///
    /// The parser cache is structured as a prefix -> suffix -> suffix -> ...
    /// -> output tree. Deduplicating common prefixes is not done every time a
    /// new cache entry is inserted because...
    ///
    /// 1. Deduplicating common prefixes is a relatively expensive operation,
    ///    with some setup costs that can be amortized.
    /// 2. Deduplication produces best results with a good knowledge of what's
    ///    being deduplicated, i.e. a relatively long list of inputs.
    /// 3. Searching a linear list is quite fast as long as the list is not too
    ///    long, whereas navigating a prefix -> suffix tree edge is a bit
    ///    expensive due to CPU cache mechanics. So deduplication only leads to
    ///    a net performance benefit when performed on longer lists.
    ///
    /// The high water mark tuning parameter dictates how big a prefix or suffix
    /// list needs to get before it gets deduplicated. Low values will lead to
    /// a smaller search space, but a more deeply nested cache structure and
    /// higher deduplication overhead. Higher values will lead to faster search
    /// in a larger search space. The right answer here is workload dependent.
    ///
    /// # Allowed values
    ///
    /// When tuning this, bear in mind that the product of this parameter by the
    /// low water mark fraction should be at least the number of unique bytes
    /// appearing your input grammar (if unsure, do not get below 128 for ASCII
    /// input and 256 for other "raw byte" input). Otherwise, the cache may need
    /// to override your setting to a more sensible value (with a warning).
    ///
    pub fn high_water_mark(mut self, mark: usize) -> Self {
        assert!(mark >= 2, "No point in deduplicating a list of <2 elements");
        self.high_water_mark = mark;
        self
    }

    /// Tune the "low water mark" parser cache deduplication parameter
    ///
    /// As discussed in the high_water_mark setting description, there is a
    /// tradeoff between maximal deduplication / minimal search space and
    /// avoidance of various kinds of run-time overheads.
    ///
    /// Since deduplication has some fixed setup overheads and can lead to bad
    /// degenerate cases (e.g. all but one suffix with one shared prefix), it
    /// makes sense to do deduplication recursively, but only up to a point
    /// (otherwise, we end up doing things like deduplicating lists of two
    /// elements, which is meaningless and will worsen cache performance).
    ///
    /// This parameter tunes how aggressive the recursive deduplication process
    /// should be, by tuning the sublist deduplication threshold as a fraction
    /// of the high water mark.
    ///
    /// # Allowed values
    ///
    /// This parameter takes values between 0.0 (exclusive) and 1.0 (inclusive),
    /// where 0.0 would mean fully recursive deduplication (if it were allowed)
    /// and 1.0 means no recursive deduplication. See also the note on allowed
    /// values for high_water_mark().
    ///
    pub fn low_water_mark_ratio(mut self, mark_ratio: f32) -> Self {
        assert!(mark_ratio > 0.0 && mark_ratio <= 1.0);
        self.low_water_mark_ratio = mark_ratio;
        self
    }

    /// Finish the parser cache building process
    pub fn build(self) -> CachingParser<Input, Output> {
        // Configure the CachingParser
        let worth_caching = self.worth_caching.unwrap_or_else(|| Box::new(|_, _| true));
        let config = CachingParserConfig {
            parser: self.parser,
            check_result: self.check_result,
            worth_caching,
            high_water_mark: self.high_water_mark,
            low_water_mark_ratio: self.low_water_mark_ratio,
        };

        // Validate low water mark configuration
        assert!(
            config.low_water_mark() > 2,
            "low_water_mark_ratio is set too low ({}) with respect to \
            high_water_mark ({}) and that leads to an excessively small low \
            water mark ({})",
            config.low_water_mark_ratio,
            config.high_water_mark,
            config.low_water_mark(),
        );

        // Build the CachingParser
        CachingParser {
            data: PrefixToOutput::with_capacity(self.high_water_mark),
            config,
        }
    }
}

// TODO: Add benchmarks
#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        cell::Cell,
        sync::atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn new() {
        // Basic mocks to test that input functions are used as expected
        static PARSER_CALLS: AtomicUsize = AtomicUsize::new(0);
        fn parser_mock(_: &str) -> Option<(&str, ())> {
            PARSER_CALLS.fetch_add(1, Ordering::Relaxed);
            Some(("", ()))
        }
        static CHECKER_CALLS: AtomicUsize = AtomicUsize::new(0);
        fn checker_mock(_: &str, _: &()) -> bool {
            CHECKER_CALLS.fetch_add(1, Ordering::Relaxed);
            true
        }
        static RETAIN_CALLS: AtomicUsize = AtomicUsize::new(0);
        fn retain_mock(_: &str, _: &()) -> bool {
            RETAIN_CALLS.fetch_add(1, Ordering::Relaxed);
            false
        }

        // Checks that should be valid no matter how the parser is configured
        let common_checks = |parser: &CachingParser<_, _>, expected_retain| {
            assert_eq!(parser.data.len(), 0);
            assert_eq!(parser.data.capacity(), parser.config.high_water_mark);

            let old_parser_calls = PARSER_CALLS.load(Ordering::Relaxed);
            let next_parser_calls = old_parser_calls + 1;
            let old_checker_calls = CHECKER_CALLS.load(Ordering::Relaxed);
            let next_checker_calls = old_checker_calls + 1;

            (parser.config.parser)("");
            assert_eq!(PARSER_CALLS.load(Ordering::Relaxed), next_parser_calls);
            assert_eq!(CHECKER_CALLS.load(Ordering::Relaxed), old_checker_calls);

            (parser.config.check_result)("", &());
            assert_eq!(PARSER_CALLS.load(Ordering::Relaxed), next_parser_calls);
            assert_eq!(CHECKER_CALLS.load(Ordering::Relaxed), next_checker_calls);

            assert_eq!((parser.config.worth_caching)("", &()), expected_retain);
            assert_eq!(PARSER_CALLS.load(Ordering::Relaxed), next_parser_calls);
            assert_eq!(CHECKER_CALLS.load(Ordering::Relaxed), next_checker_calls);
        };

        // Basic configuration
        let parser = CachingParser::new(parser_mock, checker_mock);
        common_checks(&parser, true);
        assert!(parser.config.high_water_mark >= 256);
        assert!(parser.config.low_water_mark() >= 256);

        // Custom configuration
        let parser = CachingParser::builder(parser_mock, checker_mock)
            .retention_criterion(retain_mock)
            .high_water_mark(128)
            .low_water_mark_ratio(0.75)
            .build();
        let old_retain_calls = RETAIN_CALLS.load(Ordering::Relaxed);
        common_checks(&parser, false);
        assert_eq!(RETAIN_CALLS.load(Ordering::Relaxed), old_retain_calls + 1);
        assert_eq!(parser.config.high_water_mark, 128);
        assert_eq!(parser.config.low_water_mark_ratio, 0.75);
        assert_eq!(parser.config.low_water_mark(), 96);
    }

    // Basic test parser that looks for a space and strips the beginning of the
    // string until that space if it finds it, associated check, and a way to
    // monitor how many times the parser and check were called
    fn strip_space_parser_builder() -> (
        CachingParserBuilder<str, String>,
        Rc<Cell<usize>>,
        Rc<Cell<usize>>,
    ) {
        let parse_count = Rc::new(Cell::new(0));
        let check_count = Rc::new(Cell::new(0));
        let parse_count2 = parse_count.clone();
        let check_count2 = check_count.clone();
        let builder = CachingParser::builder(
            move |input: &str| {
                parse_count2.set(parse_count2.get() + 1);
                input
                    .find(' ')
                    .map(|pos| (&input[pos..], input[..pos].into()))
            },
            move |rest, _output| {
                check_count2.set(check_count2.get() + 1);
                match rest.chars().next() {
                    Some(' ') => true,
                    _ => false,
                }
            },
        );
        (builder, parse_count, check_count)
    }
    //
    fn strip_space_parser() -> (CachingParser<str, String>, Rc<Cell<usize>>, Rc<Cell<usize>>) {
        let (builder, parse_count, check_count) = strip_space_parser_builder();
        (builder.build(), parse_count, check_count)
    }

    #[test]
    fn insert_success() {
        let (mut parser, parse_count, check_count) = strip_space_parser();
        let output = Rc::new("And".to_owned());
        assert_eq!(
            parser.get_or_insert("And then"),
            Some((" then", output.clone()))
        );
        assert_eq!(
            parser.data,
            vec![(
                "And".as_bytes().into(),
                PrefixMapping::Output(output.clone())
            )]
        );
        assert_eq!(parse_count.get(), 1);
        assert_eq!(check_count.get(), 0);
    }

    #[test]
    fn insert_failure() {
        let (mut parser, parse_count, check_count) = strip_space_parser();
        assert_eq!(parser.get_or_insert(""), None);
        assert_eq!(parser.data, PrefixToOutput::default());
        assert_eq!(parse_count.get(), 1);
        assert_eq!(check_count.get(), 0);
    }

    #[test]
    fn retrieve_success() {
        let (mut parser, parse_count, check_count) = strip_space_parser();
        parser.get_or_insert("And then");
        let output = Rc::new("And".to_owned());
        assert_eq!(
            parser.get_or_insert("And if"),
            Some((" if", output.clone()))
        );
        assert_eq!(
            parser.data,
            vec![(
                "And".as_bytes().into(),
                PrefixMapping::Output(output.clone())
            )]
        );
        assert_eq!(parse_count.get(), 1);
        assert_eq!(check_count.get(), 1);
    }

    #[test]
    fn retrieve_failure() {
        let (mut parser, parse_count, check_count) = strip_space_parser();
        parser.get_or_insert("Add to that");
        assert_eq!(parser.get_or_insert("Additionally"), None);
        assert_eq!(
            parser.data,
            vec![(
                "Add".as_bytes().into(),
                PrefixMapping::Output(Rc::new("Add".to_owned()))
            )]
        );
        assert_eq!(parse_count.get(), 2);
        assert_eq!(check_count.get(), 1);
    }

    #[test]
    fn retention_failure() {
        let (parser_builder, parse_count, check_count) = strip_space_parser_builder();
        let mut parser = parser_builder
            .retention_criterion(|input, _output| input.len() > 128)
            .build();
        assert_eq!(
            parser.get_or_insert("Something in the way she moves"),
            Some((" in the way she moves", Rc::new("Something".to_owned())))
        );
        assert_eq!(parser.data, PrefixToOutput::default());
        assert_eq!(parse_count.get(), 1);
        assert_eq!(check_count.get(), 0);
    }

    #[test]
    fn basic_deduplication() {
        // Set up the parser with a high and low watermark of 4 items
        let (parser_builder, parse_count, check_count) = strip_space_parser_builder();
        let mut parser = parser_builder
            .high_water_mark(4)
            .low_water_mark_ratio(1.0)
            .build();

        // Until the high water mark is reached, nothing happens
        parser.get_or_insert("GameObject is a Unity class");
        parser.get_or_insert("GameOn was a French journal");
        parser.get_or_insert("GameBoy is a video game console");
        let output = |s: &str| Rc::new(s.to_owned());
        let outmap = |s| PrefixMapping::Output(output(s));
        assert_eq!(
            parser.data,
            vec![
                ("GameObject".as_bytes().into(), outmap("GameObject")),
                ("GameOn".as_bytes().into(), outmap("GameOn")),
                ("GameBoy".as_bytes().into(), outmap("GameBoy")),
            ]
        );

        // Once the high water mark is reached, deduplication occurs
        // Because we use the maximal low water mark, it is not recursive here
        parser.get_or_insert("Not everything is about games");
        assert_eq!(
            parser.data,
            vec![
                (
                    "Game".as_bytes().into(),
                    PrefixMapping::Suffixes(vec![
                        ("Boy".as_bytes().into(), outmap("GameBoy")),
                        // Notice the remaining "O" duplication here
                        ("Object".as_bytes().into(), outmap("GameObject")),
                        ("On".as_bytes().into(), outmap("GameOn")),
                    ])
                ),
                ("Not".as_bytes().into(), outmap("Not")),
            ]
        );

        // No need for the parser to adjust the HWM here
        assert_eq!(parser.config.high_water_mark, 4);

        // This should not result in new parser calls
        assert_eq!(parse_count.get(), 4);
        assert_eq!(check_count.get(), 0);

        // The cache still works as before with the deduplicated layout
        assert_eq!(
            parser.get_or_insert("GameObject used to be an adventurer like you"),
            Some((" used to be an adventurer like you", output("GameObject")))
        );
        assert_eq!(parse_count.get(), 4);
        assert_eq!(check_count.get(), 1);
        assert_eq!(
            parser.get_or_insert("GameObjection could exist in Phoenix Wright"),
            Some((" could exist in Phoenix Wright", output("GameObjection")))
        );
        assert_eq!(parse_count.get(), 5);
        assert_eq!(check_count.get(), 2);
    }

    #[test]
    fn recursive_deduplication() {
        // Set up the parser with a high water mark of 4 items and a low water
        // mark of 2 items (= maximal deduplication)
        let (parser_builder, parse_count, check_count) = strip_space_parser_builder();
        let mut parser = parser_builder
            .high_water_mark(5)
            .low_water_mark_ratio(0.601)
            .build();
        assert_eq!(parser.config.low_water_mark(), 3);

        // Until we reached 4 items, nothing happens, it's the high water mark
        // that dictates when deduplication occurs.
        parser.get_or_insert("GameObject is a Unity class");
        parser.get_or_insert("GameOn was a French journal");
        parser.get_or_insert("GameBoy is a video game console");
        parser.get_or_insert("Not everything is about games");
        let output = |s: &str| Rc::new(s.to_owned());
        let outmap = |s| PrefixMapping::Output(output(s));
        assert_eq!(
            parser.data,
            vec![
                ("GameObject".as_bytes().into(), outmap("GameObject")),
                ("GameOn".as_bytes().into(), outmap("GameOn")),
                ("GameBoy".as_bytes().into(), outmap("GameBoy")),
                ("Not".as_bytes().into(), outmap("Not")),
            ]
        );

        // Above the high water mark, the deduplication algorithm gets more
        // aggressive, tying to get down to the individual matches
        parser.get_or_insert("GameBowl is not a thing (yet?)");
        assert_eq!(
            parser.data,
            vec![
                (
                    "Game".as_bytes().into(),
                    PrefixMapping::Suffixes(vec![
                        (
                            "Bo".as_bytes().into(),
                            PrefixMapping::Suffixes(vec![
                                ("wl".as_bytes().into(), outmap("GameBowl")),
                                ("y".as_bytes().into(), outmap("GameBoy")),
                            ])
                        ),
                        (
                            "O".as_bytes().into(),
                            PrefixMapping::Suffixes(vec![
                                ("bject".as_bytes().into(), outmap("GameObject")),
                                ("n".as_bytes().into(), outmap("GameOn")),
                            ])
                        ),
                    ])
                ),
                ("Not".as_bytes().into(), outmap("Not")),
            ]
        );
        assert_eq!(parse_count.get(), 5);
        assert_eq!(check_count.get(), 0);

        // No need for the parser to adjust the HWM here either
        assert_eq!(parser.config.high_water_mark, 5);
    }
}
