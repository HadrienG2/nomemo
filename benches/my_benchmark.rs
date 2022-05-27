use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use nomemo::CachingParserBuilder;

fn parse(input: &str) -> Option<(&str, ())> {
    input.find(' ').map(|pos| (&input[pos..], ()))
}
//
fn builder() -> CachingParserBuilder {
    CachingParserBuilder::new(parse, |rest, ()| match rest.chars().next() {
        Some(' ') => true,
        _ => false,
    })
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // Raw overhead of the underlying parser being wrapped
    c.bench_function("direct/pass", |b| b.iter(|| parse(black_box("A thing"))));
    c.bench_function("direct/fail", |b| b.iter(|| parse(black_box("A"))));

    // Indirection and minimal insertion overhead test: starting from an empty
    // cache, do the same parse as above, but this time with the parser being
    // wrapped into a CachingParser, with or without retention.
    c.bench_function("insert/pass/discard", |b| {
        b.iter_batched_ref(
            || builder().retention_criterion(|_, _| false).build(),
            |parser| parser.get_or_insert(black_box("A thing")),
            BatchSize::LargeInput,
        )
    });
    let setup = || builder().build();
    c.bench_function("insert/pass/cache", |b| {
        b.iter_batched_ref(
            setup,
            |parser| parser.get_or_insert(black_box("A thing")),
            BatchSize::LargeInput,
        )
    });
    c.bench_function("insert/fail", |b| {
        b.iter_batched_ref(
            setup,
            |parser| parser.get_or_insert(black_box("A")),
            BatchSize::LargeInput,
        )
    });

    // Best-case retrieval scenario: only one entry in cache, which is the one
    // being retrieved (or not). Here, we need to distinguish between two
    // retrieval failure scenarios: either the cached prefix doesn't match, or
    // it matches but the resulting estimated parse doesn't pass the check.
    let setup = || {
        let mut parser = builder().build();
        parser.get_or_insert("A thing");
        parser
    };
    c.bench_function("retrieve/pass", |b| {
        b.iter_batched_ref(
            setup.clone(),
            |parser| parser.get_or_insert(black_box("A similar thing")),
            BatchSize::LargeInput,
        )
    });
    c.bench_function("retrieve/fail/prefix", |b| {
        b.iter_batched_ref(
            setup,
            |parser| parser.get_or_insert(black_box("B sides...")),
            BatchSize::LargeInput,
        )
    });
    c.bench_function("retrieve/fail/check", |b| {
        b.iter_batched_ref(
            setup,
            |parser| parser.get_or_insert(black_box("An other thing")),
            BatchSize::LargeInput,
        )
    });

    // TODO: Then do benchmark groups + throughput numbers of these scenarios :
    //
    // TODO: Unsuccessful search in an undeduplicated list of varying length
    //       (use a pair of base64 digits + a big high water mark above the
    //       dataset size).
    //
    //       This gives an idea of the speed at which we search through a list
    //       when the overhead of checking an entry is negligible.
    //
    // TODO: Unsuccessful search in a worst-case deduplicated tree of varying
    //       depth (use sequences of "AA" of varying length + small low water
    //       mark and high water mark that is triggered as the last element gets
    //       inserted).
    //
    //       "AA" -> ""
    //            -> "AA" -> ""
    //                    -> "AA" -> ...
    //
    //       This gives an idea of the speed at which we dive through a
    //       deduplicated tree when the overhead of checking an entry is
    //       negligible, hinting at the tree vs list tradeoff.
    //
    // TODO: Unsuccessful search in a best-case deduplicated tree of fixed
    //       number of elements (say, 4096 to fit in two base64 digits) but
    //       varying arity (each layer of the tree divides the search space by
    //       N. Dataset is made of pairs of base64 digits again, but chosen to
    //       achieve the desired deduplication pattern.
    //
    //       "00" -> "00" -> "00" -> ...
    //                    -> "01" -> ...
    //                    -> "02" -> ...
    //                    -> "03" -> ...
    //            -> "01" -> ...
    //            -> "02" -> ...
    //            -> "03" -> ...
    //       "01" -> ...
    //       "02" -> ...
    //       "03" -> ...
    //
    //       This gives an idea of how much a tree can save time with respect to
    //       a list when more or less optimal balancing is achieved.
    //
    // TODO: Then find a good microbenchmark for the deduplication process
    //       itself, with varying degrees of recursion, probably using a variant
    //       of the last dataset above.
    //
    // TODO: Then put it all together with a mixed insertion/retrieval
    //       benchmark, that shows influence of the water mark tuning parameters
    //       at various hit/miss ratios.
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
