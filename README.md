# A memoization decorator for nom parsers

So, you have a nom parser that takes too much time to run, and you have done
anything in your power to speed up its internal logic. Hence it's time to move
to the last line of performance optimization, result memoization.

This crate provides a `CachingParser` that wraps a nom `Parser` and maintains a
cache of previous (input -> output) parses in order to avoid calling that parser
on inputs for which the output is already known.
