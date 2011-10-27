#!/usr/bin/perl

use strict;
use warnings;

BEGIN {
    use Test::More;
    eval { require RepRoot };
    plan skip_all => 'RepRoot is required to run math tests' if ($@);

    plan tests => 9;
}

use RepRoot;

my $package = 'HMM::Trainer';
use_ok($package);

my @methods = qw(
    train_hmm
    compute_start_probabilities
    compute_transition_probabilities
    compute_emission_probabilities
    _validate_training_set
    _get_anchor_counts
    _get_transition_counts
    _get_emission_counts
);

can_ok($package, $_) for @methods;

