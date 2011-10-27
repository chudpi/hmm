#!/usr/bin/perl

use strict;
use warnings;

BEGIN {
    use Test::More;
    eval { require RepRoot };
    plan skip_all => 'RepRoot is required to run math tests' if ($@);

    eval { require Test::Pod };
    plan skip_all => "Test::Pod 1.14 required for testing POD" if $@;

    plan tests => 1;
}

Test::Pod::pod_file_ok( "$RepRoot::ROOT/lib/HMM/Trainer.pm", 'Valid POD');
