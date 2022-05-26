// TODO: General crate docs + deny(missing_docs)

use std::rc::Rc;

// TODO: Generalize
type Input<'a> = &'a str;
type Output = ();

/// Cache of strings we've parsed before and associated parser output
pub struct CachingParser {
    /// Cached (string -> output) mappings
    data: PrefixToOutput,

    /// Cache configuration
    config: CachingParserConfig,
}
//
/// Mapping from a chunk of parser input (prefix string) to associated output or
/// possibly next strings (suffixes) eventually leading to parser output.
type PrefixToOutput = Vec<(Vec<u8>, PrefixMapping)>;
//
/// Things which a prefix string can map into
enum PrefixMapping {
    /// Direct mapping to output data
    Output(Rc<Output>),

    /// Indirect mapping via suffixes
    Suffixes(PrefixToOutput),
}
//
struct CachingParserConfig {
    /// Inner parser that we're trying to avoid calling via memoization
    ///
    /// See constructor docs for semantics.
    ///
    // TODO: Generalize to any nom parser
    parser: Box<dyn Fn(Input) -> Option<(Input, Output)>>,

    /// Check that a parser output estimated from the cache is valid
    ///
    /// See constructor docs for semantics.
    ///
    check_result: Box<dyn Fn(Input, &Output) -> bool>,

    /// Truth that a parse was complex enough to warrant memoization
    ///
    /// Receives as input the input that was parsed and the output that was
    /// produced, returns as output the truth that this is worth caching.
    ///
    worth_caching: Box<dyn Fn(Input, &Output) -> bool>,

    /// Size of a PrefixToOutput list that initiates prefix deduplication
    high_water_mark: usize,

    /// Targeted max PrefixToOutput list length after prefix deduplication,
    /// as a fraction of high_water_mark, guides deduplication recursion.
    low_water_mark_ratio: f32,
}
//
impl CachingParserConfig {
    /// Targeted max PrefixToOutput list length after prefix deduplication
    fn low_water_mark(&self) -> usize {
        (self.low_water_mark_ratio * (self.high_water_mark as f32)) as usize
    }
}
//
impl CachingParser {
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
        parser: impl Fn(Input) -> Option<(Input, Output)> + 'static,
        check_result: impl Fn(Input, &Output) -> bool + 'static,
    ) -> Self {
        Self::builder(parser, check_result).build()
    }

    /// Start building a caching parser with non-default settings
    ///
    /// See `CachingParser::new()` for more info on parameter semantics.
    ///
    pub fn builder(
        parser: impl Fn(Input) -> Option<(Input, Output)> + 'static,
        check_result: impl Fn(Input, &Output) -> bool + 'static,
    ) -> CachingParserBuilder {
        CachingParserBuilder::new(parser, check_result)
    }

    /// Find cached output associated with a string, or compute output and
    /// insert it into the cache.
    // TODO: Just implement the nom Parse trait
    pub fn get_or_insert<'input>(
        &mut self,
        input: Input<'input>,
    ) -> Option<(Input<'input>, Rc<Output>)> {
        get_or_insert_impl(&mut self.data, &mut self.config, input, input.as_ref())
    }
}

/// Mechanism to configure a CachingParser before building it
//
// See method docs for detailed member docs
pub struct CachingParserBuilder {
    /// Parser to be wrapped
    parser: Box<dyn Fn(Input) -> Option<(Input, Output)>>,

    /// Check that a parser output estimated from the cache is valid
    check_result: Box<dyn Fn(Input, &Output) -> bool>,

    /// Truth that a parse was complex enough to warrant memoization
    worth_caching: Option<Box<dyn Fn(Input, &Output) -> bool>>,

    /// Size of a PrefixToOutput list that initiates prefix deduplication
    high_water_mark: usize,

    /// Targeted max PrefixToOutput list length after prefix deduplication,
    /// as a fraction of high_water_mark.
    low_water_mark_ratio: f32,
}
//
impl CachingParserBuilder {
    /// Start configuring a CachingParser
    ///
    /// See `CachingParser::new()` for more info on parameter semantics.
    ///
    pub fn new(
        parser: impl Fn(Input) -> Option<(Input, Output)> + 'static,
        check_result: impl Fn(Input, &Output) -> bool + 'static,
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
        worth_caching: impl Fn(Input, &Output) -> bool + 'static,
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
    pub fn build(self) -> CachingParser {
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
            config.low_water_mark() >= 2,
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

/// Recursive implementation of get_or_insert targeting a cache subtree
fn get_or_insert_impl<'input>(
    tree: &mut PrefixToOutput,
    config: &mut CachingParserConfig,
    initial_input: Input<'input>,
    remaining_input: &'input [u8],
) -> Option<(Input<'input>, Rc<Output>)> {
    // Iterate through input prefixes from the current subtree
    for (prefix, mapping) in tree.iter_mut() {
        // If the prefix matches current input...
        if let Some(remaining_input) = remaining_input.strip_prefix(&**prefix) {
            match mapping {
                // ...and if we reached the final output...
                PrefixMapping::Output(o) => {
                    // ...and if the estimated parser output matches
                    // our expectations (UTF-8 remainder) and the user's...
                    if let Ok(remaining_input) = std::str::from_utf8(remaining_input) {
                        if (config.check_result)(remaining_input, &*o) {
                            // ...then we're done
                            return Some((remaining_input, o.clone()));
                        }
                    }
                }

                // If we only ended up on a subtree of suffixes...
                PrefixMapping::Suffixes(tree) => {
                    // ...then we recurse into that subtree
                    return get_or_insert_impl(tree, config, initial_input, remaining_input);
                }
            }
        }
    }

    // If no prefix matches, then there is no cached output for this input,
    // and we must run the parser
    let prefix_len = initial_input.len() - remaining_input.len();
    let (remaining_input, output) = (config.parser)(initial_input)?;

    // Does this parser result feel worth caching?
    let parsed_len = initial_input.len() - remaining_input.len();
    let parsed_input = &initial_input[..parsed_len];
    let output = if (config.worth_caching)(parsed_input, &output) {
        // If so, memoize it
        let new_suffix = parsed_input.as_bytes()[prefix_len..].into();
        let output = Rc::new(output);
        tree.push((new_suffix, PrefixMapping::Output(output.clone())));

        // Deduplcate the current subtree if high water mark is reached
        if tree.len() > config.high_water_mark {
            deduplicate(tree, config, 1.0);
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
fn deduplicate(tree: &mut PrefixToOutput, config: &mut CachingParserConfig, water_mark_ratio: f32) {
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
        let ref_byte = reference.0[0];
        while let Some(matching) = old_tree_iter.next_if(|(k, _v)| k[0] == ref_byte) {
            matches.push(matching);
        }

        // If that search yielded at least one result...
        if !matches.is_empty() {
            // ...determine the longest common prefix
            let prefix = lcp(&reference.0[..], matches.iter().map(|(k, _v)| &k[..]));

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
            if suffixes.len() > config.low_water_mark() {
                deduplicate(&mut suffixes, config, config.low_water_mark_ratio);
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
    if tree.len() > (config.high_water_mark as f32 * water_mark_ratio) as usize {
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

// TODO: Tests and benchmarks
