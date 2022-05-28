use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
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
    c.bench_function("insert/pass/cache", |b| {
        b.iter_batched_ref(
            || builder().build(),
            |parser| parser.get_or_insert(black_box("A thing")),
            BatchSize::LargeInput,
        )
    });
    c.bench_function("insert/fail", |b| {
        b.iter_batched_ref(
            || builder().build(),
            |parser| parser.get_or_insert(black_box("A")),
            BatchSize::LargeInput,
        )
    });

    // Best-case retrieval scenario: only one entry in cache, which is the one
    // being retrieved (or not). Here, we need to distinguish between two
    // retrieval failure scenarios: either the cached prefix doesn't match, or
    // it matches but the resulting estimated parse doesn't pass the check.
    let setup = || {
        let mut parser = builder()
            .retention_criterion(|input, _| input == "A")
            .build();
        parser.get_or_insert("A thing");
        parser
    };
    c.bench_function("retrieve/pass", |b| {
        b.iter_batched_ref(
            setup,
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

    // A bit of base64 infrastructure will come in handy for the next benches
    const DIGIT: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let div_round_up = |num, den| num / den + (num % den != 0) as u32;
    let num_digits = |x: usize| div_round_up(x.trailing_zeros(), 64usize.trailing_zeros());
    let write_digits = |buf: &mut String, x: usize, num_digits: u32| {
        let mut acc = x;
        for _ in 0..num_digits {
            let digit = acc % 64;
            buf.push(DIGIT[digit] as char);
            acc /= 64;
        }
    };

    // Unsuccessful search in an undeduplicated list of varying length where all
    // entries can be eliminated by looking up one single character. This gives
    // an idea of how quickly we iterate through a list of prefixes/suffixes
    // when the overhead of actually checking an entry is negligible.
    {
        let setup = |num_inputs: usize| {
            let mut parser = builder()
                .high_water_mark(10 * num_inputs)
                .low_water_mark_ratio(1.0)
                .build();
            let num_digits = num_digits(num_inputs);
            let mut buf = String::with_capacity(num_digits as usize + 1);
            for input in 0..num_inputs {
                write_digits(&mut buf, input, num_digits);
                buf.push(' ');
                parser.get_or_insert(&buf);
                buf.clear();
            }
            parser
        };
        let mut group = c.benchmark_group("search/linear");
        for num_inputs in [2, 4, 8, 16, 32, 64, 128, 256, 512, 4096] {
            group.throughput(Throughput::Elements(num_inputs as _));
            group.bench_with_input(
                BenchmarkId::from_parameter(num_inputs as u64),
                &num_inputs,
                |b, _| {
                    b.iter_batched_ref(
                        || setup(num_inputs),
                        |parser| parser.get_or_insert(black_box("##")),
                        BatchSize::LargeInput,
                    )
                },
            );
        }
    }

    // The worst-case scenario for the current deduplication algorithm is a
    // unary integer encoding where all lengths are present:
    //
    //       ""
    //       "A" -> ""
    //           -> "A" -> ""
    //                  -> "A" -> ...
    //
    // This gives an idea of the performance of navigating the prefix->suffix
    // tree layers, which together with the above benchmark illustrates the
    // tradeoff of deduplication: it reduces the search space, but makes search
    // slower, which is why we have we have water marks as a way to reduce the
    // degree of deduplication when it does more harm than good.
    {
        let setup = |tree_depth| {
            let mut parser = builder()
                .high_water_mark(tree_depth)
                .low_water_mark_ratio(3 as f32 / tree_depth as f32 + f32::EPSILON)
                .build();
            let mut buf = Vec::<u8>::with_capacity(tree_depth + 1);
            buf.push(b' ');
            for _ in 0..tree_depth {
                parser.get_or_insert(std::str::from_utf8(&buf).unwrap());
                buf.insert(buf.len() - 1, b'A');
            }
            parser
        };
        let mut group = c.benchmark_group("search/nested");
        for tree_depth in [4, 8, 16, 32, 64, 128, 256, 512, 4096] {
            group.throughput(Throughput::Elements(tree_depth as _));
            let input = std::iter::repeat('A').take(tree_depth).collect::<String>();
            group.bench_with_input(
                BenchmarkId::from_parameter(tree_depth as u64),
                &tree_depth,
                |b, _| {
                    b.iter_batched_ref(
                        || setup(tree_depth),
                        |parser| parser.get_or_insert(black_box(&input)),
                        BatchSize::LargeInput,
                    )
                },
            );
        }
    }

    // TODO: Then do benchmark groups + throughput numbers of these scenarios :
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
    //       a list when more or less optimal balancing is achieved. Try both
    //       the scenario where the search ends after navigating the prefix list
    //       and that where it needs to get to the bottom of the tree. Measure
    //       throughput wrt the number of elements in the tree (search space
    //       size) so that numbers are comparable to the above benchmarks.
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
