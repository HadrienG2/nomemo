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
// TODO: Add a construction/configuration mechanism
struct CachingParserConfig {
    /// Inner parser that we're trying to avoid calling via memoization
    // TODO: Generalize to any nom parser
    parser: Box<dyn Fn(Input) -> Option<(Input, Output)>>,

    /// Truth that a parse was complex enough to warrant memoization
    ///
    /// Receives as input the input that was parsed and the output that was
    /// produced, returns as output the truth that this is worth caching.
    ///
    worth_caching: Box<dyn Fn(Input, &Output) -> bool>,

    /// Check that a parser output estimated from the cache is valid
    ///
    /// Receives as input the estimated parse result (remaining unparsed input
    /// + expected output), returns truth that this result is valid in the sense
    /// that a direct parse on the input string wouldn't have returned a
    /// different result.
    ///
    /// This check is needed for grammars with optional terminators, e.g. in
    /// order to correctly parse "Method() const" when "Method()" is in cache.
    ///
    check_result: Box<dyn Fn(Input, &Output) -> bool>,

    /// Size of a PrefixToOutput list that initiates prefix deduplication
    ///
    /// Should be chosen such that low_water_mark_ratio * high_water_mark >= 2,
    /// otherwise we'll try to deduplicate subtrees with only one element which
    /// is stupid.
    ///
    high_water_mark: usize,

    /// Targeted max PrefixToOutput list length after prefix deduplication,
    /// as a fraction of high_water_mark.
    ///
    /// Should be in the ] 0.0, 1.0 ] range, anything too close to zero is most
    /// likely a mistake per the above constraint on high_water_mark.
    ///
    low_water_mark_ratio: f32,
}
//
impl CachingParser {
    /// Find cached output associated with a string, or compute output and
    /// insert it into the cache.
    // TODO: Just implement the nom Parse trait
    pub fn get_or_insert<'input>(
        &mut self,
        input: Input<'input>,
    ) -> Option<(Input<'input>, Rc<Output>)> {
        Self::get_or_insert_impl(&mut self.data, &mut self.config, input, input.as_ref())
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
                        return Self::get_or_insert_impl(
                            tree,
                            config,
                            initial_input,
                            remaining_input,
                        );
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
        tree: &mut PrefixToOutput,
        config: &mut CachingParserConfig,
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
        let low_water_mark = (config.high_water_mark as f32 * config.low_water_mark_ratio) as usize;
        loop {
            // Extract all entries that have 1 byte in common with the reference
            let ref_byte = reference.0[0];
            while let Some(matching) = old_tree_iter.next_if(|(k, _v)| k[0] == ref_byte) {
                matches.push(matching);
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
                if suffixes.len() > low_water_mark {
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
}

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

// TODO: Tests and benchmarks
