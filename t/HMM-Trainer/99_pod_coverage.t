#!/usr/bin/perl

use strict;
use warnings;

BEGIN {
    use Test::More;
    eval { require RepRoot };
    plan skip_all => 'RepRoot is required to run math tests' if ($@);

    eval { require Test::Pod::Coverage };
    plan skip_all => "Test::Pod::Coverage is required for testing POD coverage" if $@;

    plan tests => 1;
}
use RepRoot;

Test::Pod::Coverage::pod_coverage_ok('HMM::Trainer', 'HMM::Trainer is POD covered.');
