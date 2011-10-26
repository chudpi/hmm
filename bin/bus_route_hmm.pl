#!/usr/bin/env perl

use strict;
use warnings;

use lib '../lib';  # TODO use RepRoot.

use Data::Dumper;
use HMM::Trainer;
use Algorithm::Viterbi;

# attempt to predict on which stops along my route to work 
# the bus has most likely stopped, based on the observed 
# size of the group boarding on the bus each time it 
# stopped.  


## Training Data
# bus stops
my $entities = [
    'Washington Blvd',
    'Washington Place',
    'Barbara',
    'Venice' ,
    'Charnock', 
    'Palms',
    'Woodbine', 
    'Rose',
    'Airport', 
    'National', 
    'Ocean Park Blvd',
    'Pearl',
    'Pico',
    'Olympic',
];

# boarder/exiter sets
my $conditions = [ '0-2', '3-6' , '7+'];

my $training_set = {
    'ride01' => [
        ['0-2', 'Washington Blvd'],
        ['7+', 'Venice'],
        ['0-2', 'Palms'],
        ['0-2', 'Ocean Park Blvd'],
        ['3-6', 'Pico'],
    ],
    'ride02' => [
        ['3-6', 'Washington Place'],
        ['3-6', 'Venice'],
        ['0-2', 'Rose'],
        ['7+', 'Pico'],
        ['0-2', 'Olympic'],
    ],
    'ride03' => [
        ['0-2', 'Washington Place'],
        ['0-2', 'Barbara'],
        ['0-2', 'Venice'],
        ['0-2', 'Woodbine'],
        ['3-6', 'Pico'],
        ['0-2', 'Olympic'],
    ],
    'ride04' => [
        ['7+', 'Washington Blvd'],
        ['3-6', 'Venice'],
        ['7+', 'Pico'],
        ['3-6', 'Olympic'],
    ],
    'ride05' => [
        ['0-2', 'Washington Place'],
        ['3-6', 'Venice'],
        ['0-2', 'Woodbine'],
        ['0-2', 'Rose'],
        ['7+', 'Pico'],
        ['0-2', 'Olympic'],
    ],
    # some bad data for validation
    'bad_route' => [
        ['0-2', 'Union Station'],
    ],
    'bad_set' => [
        ['12+', 'Rose'],
    ],
};

my $hmm = HMM::Trainer->new({
    entities         => $entities,
    conditions       => $conditions,
    training_set     => $training_set,
    invert_emissions => 1,
#    debug        => 1,  # heaps of verbosity
});

print Dumper $hmm->model();


print "\n\nCalculating most likely stops given the sequence of boarder counts: (1, 4, 1, 2, 9)\n\n";

my $v = Algorithm::Viterbi->new();
$v->start($hmm->model()->{'start'});
$v->transition($hmm->model()->{'transition'});
$v->emission($hmm->model()->{'emission'});

my ($prob, $v_path, $v_prob) = $v->forward_viterbi(['0-2','3-6','0-2','7+']);

print "Prob  : $prob\n";
print "V_Path: ", Dumper($v_path);
print "V_Prob: $v_prob\n"; 
