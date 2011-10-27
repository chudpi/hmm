#!/usr/bin/perl

################################################################################
# Ensure that after training all probabilities add up to 1
# (within a small margin of error).
################################################################################
use strict;
use warnings;

BEGIN {
    use Test::More;
    eval { require YAML };
    plan skip_all => 'YAML is required to run math tests' if ($@);

    eval { require List::Util };
    plan skip_all => 'List::Util is required to run math tests' if ($@);

    eval { require RepRoot };
    plan skip_all => 'RepRoot is required to run math tests' if ($@);

    plan tests => 31;
}

use RepRoot;
use YAML;
use List::Util;

use_ok('HMM::Trainer');

sub runtests {
    # load test data
    my $data = YAML::LoadFile($RepRoot::ROOT . '/t/HMM-Trainer/training_data.yaml');

    # generate the model
    my $model = init({ data=>$data, train_on_init=>1 })->model();
    test_sum_anchor( $model->{'start'} );
    test_sum_transitions( $model->{'transition'} );
    test_sum_emissions( $model->{'emission'} );

    return;
}

## Test Subs ##

sub test_sum_anchor {
    my $anchor_hsh = shift;

    # all start probabilities should add up to roughly 1
    ok( 
        adds_up_to_roughly_one( values %$anchor_hsh ) == 1,
        'Anchor probabilities add up to roughly 1'
    );

    return;
}


sub test_sum_transitions {
    my $transition_hsh = shift;;
    
    # transition probabilities for entities which did transition into other 
    # entities should add up to roughly 1.
    # those, which did not, should add up to exactly 0.
    while ( my ($entity, $transition_entity_hsh) = each %$transition_hsh ) {
        my $verdict = adds_up_to_roughly_one ( values %$transition_entity_hsh );
        if ( $verdict == 0 ) {
            ok (1, "'$entity' does not transition into any other entity in the training set.");
        }
        else {
            ok( 
                adds_up_to_roughly_one( values %$transition_entity_hsh ) == 1,
                "'$entity' transition probabilities add up to roughly 1."
            );
        }
    }
}


sub test_sum_emissions {
    my $emission_hsh = shift;;
    
    # emission probabilities for entities emitted by at least one observed condition
    # should add up to roughtly 1.
    # those not emitted by any conditions should add up to exactly 0.
    while ( my ($entity, $condition_hsh) = each %$emission_hsh ) {
        my $verdict = adds_up_to_roughly_one ( values %$condition_hsh );
        if ( $verdict == 0 ) {
            ok (1, "'$entity' does not get emitted from any training set condition.");
        }
        else {
            ok( 
                adds_up_to_roughly_one( values %$condition_hsh ) == 1,
                "'$entity' emission probabilities add up to roughly 1."
            );
        }
    }

}


## Support Subs ##

sub init {
    my $args = shift;

    my $hmm = HMM::Trainer->new({
        %{ $args->{'data'} }, 
        train => $args->{'train_on_init'},
    });
    isa_ok($hmm, 'HMM::Trainer');

    return $hmm;
}

# returns:
#  1 - if list values add up to 1 within a small margin of error
#  0 - if list values add up to exactly 0
# -1 - if list values add up to a non-zero value not within
#      the margin of error
sub adds_up_to_roughly_one {
    my @floats = @_;
    my $margin_of_error = 0.00001;

    my $sum = List::Util::sum(@floats);
    return 1 if ($sum == 1.0);
    return 1 if ($sum < 1.0 && $sum + $margin_of_error >= 1.0);
    return 1 if ($sum > 1.0 && $sum - $margin_of_error <= 1.0);
    return 0 if ($sum == 0.0);
    return -1;
}


## __main__ ##
runtests();
