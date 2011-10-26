#!/usr/bin/perl

use strict;
use warnings;

BEGIN {
    use Test::More;
    eval { require RepRoot };
    plan skip_all => 'RepRoot is required to run dependency tests.' if($@);
    
    plan tests => 6;
}

BEGIN {
    # Make sure HMM::Trainer exists and compiles.
    use_ok( 'HMM::Trainer' );
}

BEGIN {
    diag( "Testing HMM::Trainer $HMM::Trainer::VERSION, Perl $], $^X" );
}

BEGIN {    
    # Make sure that the modules direct dependies are met
    use_ok('Moose');
    use_ok('Carp');
    use_ok('Data::Dumper');
    use_ok('List::MoreUtils');
}


ok( defined $HMM::Trainer::VERSION, 'module is versioned' );
