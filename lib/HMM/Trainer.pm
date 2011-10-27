package HMM::Trainer; {
    
    use strict;
    use warnings;

    # MOOOOOOOOOOSE!
    use Moose;

    # Helpers
    use Carp;
    use Data::Dumper;
    use List::MoreUtils 'none';

    our $VERSION = 0.1.0;

    # Constructor arguments required for training the model
    has 'entities'     => (is => 'rw', isa => 'ArrayRef', required => 1 ); # hidden states
    has 'conditions'   => (is => 'rw', isa => 'ArrayRef', required => 1 ); # observable conditions
    has 'training_set' => (is => 'rw', isa => 'HashRef',  required => 1 ); # training data
    
    # Optional external hashref within which to store the model
    has 'model'        => (is => 'rw', isa => 'HashRef',  default  => sub{ {} }); # probability model (output HMM)

    # Optional flags
    has 'train'            => (is => 'ro', isa => 'Bool',   default => 1 ); # train by default
    has 'invert_emissions' => (is => 'ro', isa => 'Bool',   default => 0 ); # don't invert by default
    has 'verbose'      => (is => 'ro', isa => 'Bool',   default => 0 ); # print some info messages
    has 'debug'        => (is => 'ro', isa => 'Bool',   default => 0 ); # print heaps of dumps to stdout


    # Moosey object instance setup
    sub BUILD {
        my ($self, $args) = @_;

        # required object variables
        $self->entities(     $args->{'entities'    } ); 
        $self->conditions(   $args->{'conditions'  } ); 
        
        # expects an HashRef of observation sets (anonymous arrays).
        # each observation within a set is an annonymous array of 
        # observed [ condition, entity ] pairs. 
        $self->training_set( $args->{'training_set'} );
        
        # get training
        $self->train_hmm() if $self->train();;

        return;
    }

    
    ### the guts ### 

    # supervises model training activities
    sub train_hmm {
        my $self = shift;

        $self->_validate_training_set();

        $self->compute_start_probabilities();
        $self->compute_transition_probabilities();
        $self->compute_emission_probabilities();

        # debug: dump the model to stdout
        print Dumper($self->model()) if $self->debug();
        return;
    }

    
    sub compute_start_probabilities {
        my $self = shift;

        print "Computing Start Probabilities.\n" if $self->verbose();

        my ( $anchor_counts, $anchor_total ) = $self->_get_anchor_counts();
    
        # calculate start probability for each entity.
        # the anchor probability for an entity is:
        #    number of entity anchors / total number of anchors.
        my $anchor_probs = {};
        while ( my($entity, $count) = each %$anchor_counts ) {
            if ( $anchor_total ) {
                $anchor_probs->{$entity} = $count/$anchor_total;
            }
            else {
                croak "Anchor counts add up to 0.  Unable to generate probabilities.\n" . $!;
            }
        }
        
        # add the start probability hash to the model.
        $self->model()->{start} = $anchor_probs;
        return;
    }


    sub compute_transition_probabilities {
        my $self = shift;

        print "Computing Transition Probabilities.\n" if $self->verbose();
        my ($transition_counts, $prior_totals) = $self->_get_transition_counts();

        # calculate the probabilities of each entity transitioning into each entity
        # (including itself) and build the matrix of transition probabilities.
        my $trans_probs = {};
        while ( my( $prior, $trans_entities ) = each %$transition_counts ) {
            while ( my ( $trans_entity, $trans_count ) = each %$trans_entities ) {
                if ( $prior_totals->{$prior} ) {
                    $trans_probs->{$prior}->{$trans_entity} = $trans_count/$prior_totals->{$prior};
                }
                else {
                    $trans_probs->{$prior}->{$trans_entity} = 0;
                }
            }
        }

        # add the transition probability matrix to the model
        $self->model()->{'transition'} = $trans_probs;

        return;
    }


    sub compute_emission_probabilities {
        my $self = shift;

        print "Computing Emission Probabilities.\n" if $self->verbose();
        my ($emission_counts, $entity_totals) = $self->_get_emission_counts();

        # calculate the probabilities of each entity getting emitted from 
        # each observable condition.
        my $emission_probs = {};
        while ( my ($entity, $condition_counts) = each %$emission_counts ) {
            while ( my ($condition, $count) = each %$condition_counts ) {
                my $emission_probability = undef;
                if ( $entity_totals->{$entity} ) {
                    $emission_probability = $count/$entity_totals->{$entity};
                }
                else {
                    $emission_probability = 0;
                }
                
                # optionally invert the nested emission hash keys from
                # emission -> condition to condition -> emission.
                #   useful for feeding the model into Algorithm::Viterbi, which for some 
                # reason likes its emission hash keyed on conditions rather than entities.
                if ( $self->invert_emissions() ) {
                    $emission_probs->{$condition}->{$entity} = $emission_probability;
                }
                else
                {
                    $emission_probs->{$entity}->{$condition} = $emission_probability;
                }
            }
        }

        # add emission probabilities to the model
        $self->model()->{'emission'} = $emission_probs;

        return;
    }


    ### SUPPORTING SUBROUTINES ###
    
    sub _validate_training_set {
        my $self = shift;
        my $training_set = $self->training_set();

        # identify bad observations in the training set.
        my @bad_observations = ();
        while ( my ($ts_key, $observations) = each %$training_set ) {
            for my $obs (@$observations) {
                my ($cond, $entity) = @$obs;
                if ( none { $entity eq $_ } @{ $self->entities() } ) {
                    print "WARNING: removing observation '$ts_key' from the training set "
                        . "because it contains an unknown entity '$entity.'\n" 
                        if $self->debug();
                    push @bad_observations, $ts_key;
                    last;
                }
                if ( none { $cond eq $_ } @{ $self->conditions() } ) {
                    print "WARNING: removing observation '$ts_key' from the training set "
                        . "because it contains an unknown condition '$cond.'\n" 
                        if $self->debug();
                    push @bad_observations, $ts_key;
                    last;
                }
            }
        }

        # remove offending observations from the training set
        delete $training_set->{$_} for @bad_observations;

        return;
    }


    sub _get_anchor_counts {
        my $self = shift;
        my $training_set = $self->training_set();
        my $entities = $self->entities();
        my $conditions = $self->conditions();
        my $anchor_counts = {};
        my $anchor_total = 0;


        # initialize all anchor counts to 0
        for my $entity ( @$entities ) {
            $anchor_counts->{$entity} = 0;
        }

        # count anchors by condition for each entity
        for my $observations ( values %$training_set ) {
            $anchor_counts->{$observations->[0][1]}++;
            $anchor_total++;
        }

        return ( $anchor_counts, $anchor_total );
    }


    sub _get_transition_counts {
        my $self = shift;
        my $training_set = $self->training_set();
        my $entities = $self->entities();
        my $trans_counts = {};
        my $prior_totals = {};

        # initialize all transition counts to 0
        for my $prior ( @$entities ) {
            for my $trans_entity ( @$entities ){
                $trans_counts->{$prior}->{$trans_entity} = 0;
            }
        }

        # count transitions
        for my $observations ( values %$training_set ) {
            for my $idx ( 0 .. (@$observations-1) ) {
                last if $idx >= (@$observations-1); # because there's nothing more to transition to.
                #                     prior                             transition-into
                $trans_counts->{ $observations->[$idx][1] }->{ $observations->[$idx+1][1] }++;
            }
        }
        
        # count prior totals
        for my $entity ( keys(%$trans_counts) ) {
            $prior_totals->{$entity} = 0;
            for my $trans_count ( values %{ $trans_counts->{$entity} } ) {
                $prior_totals->{$entity} += $trans_count;
            }
        }

        return ($trans_counts,$prior_totals);

    }
    
    
    sub _get_emission_counts {
        my $self = shift;
        my $training_set = $self->training_set();
        my $entities = $self->entities();
        my $conditions = $self->conditions();
        my $emission_counts = {};
        my $entity_totals = {};

        # initialize all emission counts to 0
        for my $entity ( @$entities ) {
            for my $condition ( @$conditions ) {
                $emission_counts->{$entity}->{$condition} = 0;
            }
        }
        
        # count emissions
        for my $observations ( values %$training_set ) {
            for my $observation ( @$observations ) {
                $emission_counts->{ $observation->[1] }->{ $observation->[0] }++;
            }
        }

        # count entity totals
        for my $entity ( keys %$emission_counts ) {
            for my $condition_count ( values %{ $emission_counts->{$entity} } ) {
                $entity_totals->{$entity} += $condition_count;
            }
        }

        return ($emission_counts, $entity_totals);
    }

}
1;

__END__

=head1 NAME

HMM::Trainer - Train a Hidden Markov model on multiple observation sets

=head1 VERSION

Version 0.1.0

=head1 SYNOPSIS

Generates a Hidden Markov Model trained on multiple sets of 
[condition, entity] observations.

Constructor method arguments documented under METHODS below.
    
    use HMM::Trainer;
    
    my $hmm = HMM::Trainer->new({
        # Required for training
        entities     => $entity_arrayref,       # entities you want to extract (hidden states) 
        conditions   => $condition_arrayref,    # observable conditions (Markov property states)
        training_set => $training_set_hashref,  # condition-entity sequences observed in the past
        
        # Optional boolean flags
        train            => 1, # train on init
        verbose          => 1, # tell me what you're doing
        debug            => 1, # tell me EVERYTING
        invert_emissions => 1, # invert emission hash so it
                               # works with Algorithm::Viterbi
    });
    my $trained_model = $hmm->model();

    
=head1 DESCRIPTION

This module trains a Hidden Markov Model using multiple sets of past observations of sequences of occurrences in preparation for decoding using the Viterbi algorithm.

The Algorithm::Viterbi CPAN module is capable of training the model, but only on a single sequence of past observations.  This a bit of a problem when you have multiple observations of relatively short sequence, because you must either opt to train the model on only one of your training set observation sequences (Algorithm::Viterbi::train() does not augment, but overwrites the model each time it is called) - which in case of short sequences will most likely be insufficient to train a statistically sound model, or to flatten your training set by removing the boundaries between the observed sequences - in which case the transition probabilities will be skewed because the first observation of the following sequence will be persumed to have a Markov property with the last observation of the previous sequence, while there isn't any.

HMM::Trainer ameliorates this dilemma by respecting sequence boundaries.

Furthermore, by respecting sequence boundaries HMM::Trainer achieves significantly more accurate anchor (start) probabilities.

=head1 METHODS

=head2 new

Constructor.  

Required Arguments: Needed to train the model

=over 4

=item * entities

An array reference containing a list of scalars, each representing a unique hidden state ( extraction category);

=back

=over 4

=item * conditions

An array reference containing a list of possible conditions (observable states).

=back

=over 4

=item * training_set

A hash reference containing a list of observed sequences keyed on a unique sequence id. The values of the hash - the sequence - is an anonymous array containing nested arrays of observed [ condition, entity ] pairs.  In other words, the structure looks as follows:
  $training_set_hashref = {
      sequence_1 => [
          [ condition_1, entity_1 ],
          [ condition_2, entity_2 ],
          [         ...         ],
          [ condition_n, entity_n ],
      ],
      sequence_2 => [
          [ condition_1, entity_1 ],
          [ condition_2, entity_2 ],
          [         ...         ],
          [ condition_n, entity_n ],
      ],
      sequence_n => [
          [ condition_1, entity_1 ],
          [ condition_2, entity_2 ],
          [         ...         ],
          [ condition_n, entity_n ],
      ],
  }

If this is a bit difficult to follow, perhaps an example will help.  Consider that you are  training a model whereby you wish to decode the probable sequence of stops a bus made along a known route given a sequence of the number of passengers ( groupped into selected  sets ) that boarded the at each stop.  The training set would comprise of data you recorded during your previous trips when you forgot to bring a book and needed something else to keep you occupied.  Your training set schema would be:
  $training_set = {
      'trip_no' => [
          [ 'number of passengers boarded', 'name of the bus stop' ],
      ],
  };

And the training set containign the data you've collected would look something like:
  $training_set_hashref = {
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
  }

As of the time of this writing you, dear user, are tasked with supplying the training set in the correct format.  Luckily the Author, yours truly, likes YAML way more than he would ever admit to anyone at a coctail party, and is already working on a more digestable way of injesting the trainig sets (there is also a high priority TODO to allow streaming in training sets too large to load into memory).

=back

Optional Arguments:

=over 4

=item * model

A hash reference to hold the trained module.  Default: anonymous hash.

=back

Optional Flags:

=over 4

=item * train

Instructs the constructor to train the model as part of constructor initialization.  Default: True.

=back

=over 4

=item * invert_emissions

Instructs the trainer to invert the nesting of entities and conditions as keys in the emission hash.  By default the module nests entities as outer hash keys and conditions as inner hash keys - as it is described in the Wikipedia "Viterbi Algorithm" page, http://en.wikipedia.org/wiki/Viterbi_algorithm .  This option is provided as convenience for easy feeding the generated model into the Algorithm::Viterbi CPAN module, which for reasons undefined inverts the nesting.

=back

=over 4

=item * verbose

Turns on the Trainer's modest verbosity.

=back

=over 4

=item * debug

Turns on the Trainer's debug mode causing it to dump structures and sanity checks along the way.

=back

=head2 BUILD

=over 4

Moose's preferred way of instantiating objects.

=back

=head2 train_hmm

=over 4

Controller method which manages the training of the start, transition, and emission probabilities.

=back

=head2 compute_start_probabilities

=over 4

Computes and populates the model with start probabilities.

=back

=head2 compute_transition_probabilities

=over 4

Computes and populates the model with transition probabilities.

=back

=head2 compute_emission_probabilities

=over 4

Computes and populates the model with transition probabilities.

=back


Internal Methods

=head2 _validate_training_set

=over 4

Verifies the soundness of the training set.
Removes sequences containing unknown entities or conditions.

=back

=head2 _get_anchor_counts

=over 4

Traverses the training set and collects entity anchor counts.

=back

=head2 _get_transition_counts

=over 4

Traverses the training set and collects entity transition counts.

=back

=head2 _get_emission_counts

=over 4

Traverses the training set and collects entity emission countsper condition..

=back

=head2 model train invert_emissions debug verbose

=over 4

Accessors (provided by Moose):

=back

=head1 TODOs

Accept a stream or iterable generator for large training sets to protect memory.

Introduce an option to lump unknown entities in a training set under 'other' entity.


=head1 AUTHOR

Pete Chudykowski, C<< <pete at chudpi.org> >>

=head1 COPYRIGHT

This code is free software.  Use, modify and distribute as you please and at your own risk.

=cut
