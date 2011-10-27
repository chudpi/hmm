#!/usr/bin/perl

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

    plan tests => 50;
}

use_ok('HMM::Trainer');

sub runtests {
    # load test data
    my $data = YAML::LoadFile($RepRoot::ROOT . '/t/HMM-Trainer/training_data.yaml');

    # test with no training at construction time
    my $hmm = init({ data=>$data, train_on_init=>0 });
    test_data_subs( $hmm, $data );
    test_crunch_subs( $hmm );
    test_train_hmm( $hmm );
    undef $hmm;

    # retest with training at construction
    $hmm = init({ data=>$data, train_on_init=>1 });
    test_data_subs( $hmm, $data );
    test_crunch_subs( $hmm );
}

sub init {
    my $args = shift;

    my $hmm = HMM::Trainer->new({
        %{ $args->{'data'} }, 
        train => $args->{'train_on_init'},
    });
    isa_ok($hmm, 'HMM::Trainer');

    return $hmm;
}

sub test_data_subs {
    my ($hmm, $data) = @_;
    
    test_verify_data_load($hmm, $data);
    test__validate_training_set($hmm, $data);
}

sub test_crunch_subs {
    my $hmm = shift;

    test__get_anchor_counts($hmm);
    test__get_transition_counts($hmm);
    test__get_emission_counts($hmm);
    test_compute_start_probabilities($hmm);
    test_compute_transition_probabilities($hmm);
    test_compute_emission_probabilities($hmm);
}


sub test_verify_data_load {
    my ($hmm, $data) = @_;

    is_deeply( $hmm->entities(), $data->{'entities'}, 'Entities loaded.');
    is_deeply( $hmm->conditions(), $data->{'conditions'}, 'Conditions loaded.');
    is_deeply( $hmm->training_set(), $data->{'training_set'}, 'Training set loaded.');
}


sub test__validate_training_set {
    my ($hmm, $data) = @_;

SKIP: {
        skip("Already trained.", 2) if( $hmm->train() );
        my $wrong_route_memcpy = $data->{'training_set'}{'wrong_route'};
        my $bad_set_memcpy = $data->{'training_set'}{'bad_set'};
        $hmm->_validate_training_set();
        ok( $wrong_route_memcpy && !exists $hmm->training_set()->{'wrong_route'}, 'Observation set with malformed entity has been removed from the training set.');
        ok( $bad_set_memcpy && !exists $hmm->training_set()->{'bad_set'}, 'Observation set with malformed condition has been removed from the training set.');
    }
}


sub test__get_anchor_counts {
    my $hmm = shift;

    my ($anchor_counts, $anchor_total) = $hmm->_get_anchor_counts();

    ok($anchor_total == 5, 'Anchor total ok.');
    # spot-check anchor counts
    ok($anchor_counts->{'Washington Blvd'} == 2, 'Entity anchor count ok.');
    ok($anchor_counts->{'Pico'} == 0, 'Entity anchor count ok.');
    ok($anchor_counts->{'Washington Place'} == 3, 'Entity anchor count ok.');
}


sub test__get_transition_counts {
    my $hmm = shift;
    my ($transition_counts, $prior_totals) = $hmm->_get_transition_counts();

    # spot-check totals
    ok($prior_totals->{'Rose'} == 2, 'Rose transition total ok.');
    ok($prior_totals->{'Venice'} == 5, 'Venice transition total ok.');
    # spot-check transition counts

    ok($transition_counts->{'Venice'}{'Rose'} == 1, 'Transition count from Venice to Rose ok.');
    ok($transition_counts->{'Pico'}{'Olympic'} == 4, 'Transition count from Pico to Olympic ok.');
}


sub test__get_emission_counts {
    my $hmm = shift;
    my ($emission_counts, $emission_totals) = $hmm->_get_emission_counts();

    # spot-check totals
    ok($emission_totals->{'Olympic'} == 4, 'Olympic emission total ok.');
    ok($emission_totals->{'Venice'} == 5, 'Venice emission total ok.');

    # spot-check emission counts
    ok($emission_counts->{'Venice'}{'0-2'} == 1, 'Emission of Venice given 0-2 boarders ok.');
    ok($emission_counts->{'Pico'}{'7+'} == 3, 'Emission of Pico given 7+ boarders ok.');
}


sub test_compute_start_probabilities {
    my $hmm = shift;
    $hmm->compute_start_probabilities();

    # spot-check start probabilities
    ok($hmm->model()->{'start'}{'Venice'} == 0, 'Start probability for Venice ok.');
    ok($hmm->model()->{'start'}{'Washington Place'} == .6, 'Start probability for Washington Place ok.');
}


sub test_compute_transition_probabilities {
    my $hmm = shift;
    $hmm->compute_transition_probabilities();

    # spot-check transition probabilities
    ok($hmm->model()->{'transition'}{'Venice'}{'Rose'} == .2, 'Venice->Rose transition probability ok.');
    ok($hmm->model()->{'transition'}{'Pico'}{'Olympic'} == 1, 'Pico->Olympic transition probability ok.');
}


sub test_compute_emission_probabilities {
    my $hmm = shift;
    $hmm->compute_emission_probabilities();

    # spot-check emission probabilities
    ok($hmm->model()->{'emission'}{'Venice'}{'7+'} == .2, 'Probability of emitting Venice given 7+ ok.');
    ok($hmm->model()->{'emission'}{'Rose'}{'3-6'} == 0, 'Probability of emitting Rose given 3-6 ok.');
}


sub test_train_hmm {
    my $hmm = shift;
    
SKIP: {
        eval { require Test::Exception };
        skip("Test::Exception is required to run this test", 1) if($@);
        skip("Already trained.", 1) if( $hmm->train() );
        Test::Exception::lives_ok( sub { $hmm->train_hmm() }, 'train_hmm lives' );
    }
}


## __main__ ##
runtests();
