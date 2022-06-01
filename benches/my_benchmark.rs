use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use nom::{IResult, Parser};
use nomemo::CachingParserBuilder;

fn parse(input: &str) -> IResult<&str, (), ()> {
    use nom::{
        character::complete::satisfy,
        combinator::{recognize, value},
        multi::many0_count,
    };
    value((), recognize(many0_count(satisfy(|c| c != ' ')))).parse(input)
}
//
fn builder() -> CachingParserBuilder<str, (), ()> {
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
            |parser| parser.parse(black_box("A thing")),
            BatchSize::LargeInput,
        )
    });
    c.bench_function("insert/pass/cache", |b| {
        b.iter_batched_ref(
            || builder().build(),
            |parser| parser.parse(black_box("A thing")),
            BatchSize::LargeInput,
        )
    });
    c.bench_function("insert/fail", |b| {
        b.iter_batched_ref(
            || builder().build(),
            |parser| parser.parse(black_box("A")),
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
        parser.parse("A thing");
        parser
    };
    c.bench_function("retrieve/pass", |b| {
        b.iter_batched_ref(
            setup,
            |parser| parser.parse(black_box("A similar thing")),
            BatchSize::LargeInput,
        )
    });
    c.bench_function("retrieve/fail/prefix", |b| {
        b.iter_batched_ref(
            setup,
            |parser| parser.parse(black_box("B sides...")),
            BatchSize::LargeInput,
        )
    });
    c.bench_function("retrieve/fail/check", |b| {
        b.iter_batched_ref(
            setup,
            |parser| parser.parse(black_box("An other thing")),
            BatchSize::LargeInput,
        )
    });

    // A bit of base64 infrastructure will come in handy for the next benches
    const DIGIT: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let div_round_up = |num, den| num / den + (num % den != 0) as u32;
    let num_digits = |x: usize, base: usize| {
        assert!(base >= 2);
        div_round_up(x.trailing_zeros(), base.trailing_zeros())
    };
    let write_digits = |buf: &mut String, x: usize, num_digits: u32, base: usize| {
        assert!(base >= 2 && base <= 64);
        let mut acc = x;
        for _ in 0..num_digits {
            let digit = acc % base;
            buf.push(DIGIT[digit] as char);
            acc /= base;
        }
    };

    // Unsuccessful search in an undeduplicated list of varying length where all
    // entries can be eliminated by looking up one single character. This gives
    // an idea of how quickly we iterate through a list of prefixes/suffixes
    // when the overhead of actually checking an entry is negligible (otherwise
    // we're just measuring the cost of strcmp, which is not super interesting)
    {
        let setup = |num_items: usize| {
            let mut parser = builder()
                .high_water_mark(10 * num_items)
                .low_water_mark_ratio(1.0)
                .build();
            let num_digits = num_digits(num_items, 64);
            let mut buf = String::with_capacity(num_digits as usize + 1);
            for input in 0..num_items {
                write_digits(&mut buf, input, num_digits, 64);
                buf.push(' ');
                parser.parse(&buf);
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
                        |parser| parser.parse(black_box("##")),
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
                parser.parse(std::str::from_utf8(&buf).unwrap());
                buf.insert(buf.len() - 1, b'A');
            }
            parser
        };
        let mut group = c.benchmark_group("search/nested");
        for tree_depth in [4, 8, 16, 32, 64, 128, 256, 512] {
            group.throughput(Throughput::Elements(tree_depth as _));
            let input = std::iter::repeat('A').take(tree_depth).collect::<String>();
            group.bench_with_input(
                BenchmarkId::from_parameter(tree_depth as u64),
                &tree_depth,
                |b, _| {
                    b.iter_batched_ref(
                        || setup(tree_depth),
                        |parser| parser.parse(black_box(&input)),
                        BatchSize::LargeInput,
                    )
                },
            );
        }
    }

    // Successful search in a best-case perfectly balanced deduplicated tree:
    //
    //       "0" -> "0" -> "0" -> ...
    //                  -> "1" -> ...
    //                  -> "2" -> ...
    //                  -> "3" -> ...
    //           -> "1" -> ...
    //           -> "2" -> ...
    //           -> "3" -> ...
    //       "1" -> ...
    //       "2" -> ...
    //       "3" -> ...
    //
    // In this benchmark, we search for the last element of a perfectly balanced
    // tree of varying arity (from binary, to 64-ary). Comparing this to
    // search/linear/4096 gives an idea of why we bother with deduplication, and
    // the perf dependance on tree arity is also interesting for those wondering
    // about the tree depth vs list length tradeoff.
    {
        const NUM_ITEMS: usize = 4096;
        let setup = |arity: usize| {
            let mut parser = builder()
                .high_water_mark(NUM_ITEMS)
                .low_water_mark_ratio(3 as f32 / NUM_ITEMS as f32 + f32::EPSILON)
                .build();
            let num_digits = num_digits(NUM_ITEMS, arity);
            let mut buf = String::with_capacity(num_digits as usize + 1);
            for input in 0..NUM_ITEMS {
                write_digits(&mut buf, input, num_digits, arity);
                buf.push(' ');
                parser.parse(&buf);
                buf.clear();
            }
            parser
        };
        let mut group = c.benchmark_group("search/tree");
        // For those analyzing benchmark results, 4096 is...
        // - 12 digits in base 2
        // - 6 digits in base 4
        // - 4 digits in base 8
        // - 3 digits in base 16
        // - 2 digits in base 64
        for arity in [2, 4, 8, 16, 64] {
            group.throughput(Throughput::Elements(NUM_ITEMS as _));
            let num_digits = num_digits(NUM_ITEMS, arity);
            let mut input = String::with_capacity(num_digits as usize);
            write_digits(&mut input, NUM_ITEMS - 1, num_digits, arity);
            group.bench_with_input(BenchmarkId::from_parameter(arity as u64), &arity, |b, _| {
                b.iter_batched_ref(
                    || setup(arity),
                    |parser| parser.parse(black_box(&input)),
                    BatchSize::LargeInput,
                )
            });
        }
    }

    // Finally, we measure the performance of the deduplication algorithm in
    // various configurations.
    {
        let setup = |num_prefixes: usize, num_suffixes: usize| {
            assert!(num_prefixes <= 64 && num_suffixes <= 64);
            let num_items = num_prefixes * num_suffixes;
            let mut parser = builder()
                .high_water_mark(num_items)
                .low_water_mark_ratio(1.0)
                .build();
            let mut buf = String::with_capacity(3);
            for prefix in 0..num_prefixes {
                write_digits(&mut buf, prefix, 1, 64);
                for suffix in 0..num_suffixes {
                    // Do not insert the last element during setup...
                    if (prefix == num_prefixes - 1) && (suffix == num_suffixes - 1) {
                        break;
                    }
                    write_digits(&mut buf, suffix, 1, 64);
                    buf.push(' ');
                    parser.parse(&buf);
                    buf.pop();
                    buf.pop();
                }
                buf.clear();
            }
            parser
        };
        let mut group = c.benchmark_group("deduplicate");
        for num_prefixes in [2, 4, 8, 16, 32, 64] {
            for num_suffixes in [2, 4, 8, 16, 32, 64] {
                let num_items = num_prefixes * num_suffixes;
                group.throughput(Throughput::Elements(num_items as _));
                // ...for the last element will be inserted here
                let mut input = String::with_capacity(3);
                write_digits(&mut input, num_prefixes - 1, 1, 64);
                write_digits(&mut input, num_suffixes - 1, 1, 64);
                input.push(' ');
                group.bench_function(
                    format!("{num_prefixes}prefixes/{num_suffixes}suffixes"),
                    |b| {
                        b.iter_batched_ref(
                            || setup(num_prefixes, num_suffixes),
                            |parser| parser.parse(black_box(&input)),
                            BatchSize::LargeInput,
                        )
                    },
                );
            }
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
