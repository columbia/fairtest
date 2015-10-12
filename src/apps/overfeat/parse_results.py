#!/usr/bin/python

import os, sys
import random

infile = 'results_black.txt'
infile2 = 'results_white.txt'
outfile = 'data.txt'

#monkey_labels = ['squirrel monkey', 'spider monkey', 'howler monkey', 'proboscis monkey', 'chimpanzee', 'orangutan', 'gorilla', 'macaque', 'patas', 'gibbon', 'siamang', 'marmoset', 'langur', 'capuchin', 'titi', 'colobus', 'guenon', 'baboon']

#dog_labels = ["Affenpinscher", "Afghan hound", "African hunting dog", "Airedale", "American Staffordshire terrier", "Appenzeller", "Australian terrier", "Basenji", "Basset", "Beagle", "Bedlington terrier", "Bernese mountain dog", "Black-and-tan coonhound", "Blenheim spaniel", "Bloodhound", "Bluetick", "Border collie", "Border terrier", "Borzoi", "Boston bull", "Bouvier des Flandres", "Boxer", "Brabancon griffon", "Briard", "Brittany spaniel", "Bull mastiff", "Cairn", "Cardigan", "Chesapeake Bay retriever", "Chihuahua", "Chow", "Clumber", "Cocker spaniel", "Collie", "Curly-coated retriever", "Dandie Dinmont", "Dhole", "Dingo", "Doberman", "English foxhound", "English setter", "English springer", "EntleBucher", "Eskimo dog", "Flat-coated retriever", "French bulldog", "German shepherd", "German short-haired pointer", "Giant schnauzer", "Golden retriever", "Gordon setter", "Great Dane", "Great Pyrenees", "Greater Swiss Mountain dog", "Groenendael", "Ibizan hound", "Irish setter", "Irish terrier", "Irish water spaniel", "Irish wolfhound", "Italian greyhound", "Japanese spaniel", "Keeshond", "Kelpie", "Kerry blue terrier", "Komondor", "Kuvasz", "Labrador retriever", "Lakeland terrier", "Leonberg", "Lhasa", "Malamute", "Malinois", "Maltese dog", "Mexican hairless", "Miniature pinscher", "Miniature poodle", "Miniature schnauzer", "Newfoundland", "Norfolk terrier", "Norwegian elkhound", "Norwich terrier", "Old English sheepdog", "Otterhound", "Papillon", "Pekinese", "Pembroke", "Pomeranian", "Pug", "Redbone", "Rhodesian ridgeback", "Rottweiler", "Saint Bernard", "Saluki", "Samoyed", "Schipperke", "Scotch terrier", "Scottish deerhound", "Sealyham terrier", "Shetland sheepdog", "Shih-Tzu", "Siberian husky", "Silky terrier", "Soft-coated wheaten terrier", "Staffordshire bullterrier", "Standard poodle", "Standard schnauzer", "Sussex spaniel", "Tibetan mastiff", "Tibetan terrier", "Toy poodle", "Toy terrier", "Vizsla", "Walker hound", "Weimaraner", "Welsh springer spaniel", "West Highland white terrier", "Whippet", "Wire-haired fox terrier", "Yorkshire terrier"]

monkey_labels = []
dog_labels = []

with open(outfile,"w") as outf:
    outf.write('Race\tLabels')
    with open(infile) as inf:
        lines = inf.readlines()
        label_list = []
        
        for idx, line in enumerate(lines):
            labels = ' '.join(line.split(' ')[0:-1])
            if ',' in labels:
                label = labels.split(',')[0]
            else:
                label = labels
            if label in monkey_labels:
                label = 'monkey'
            if label in dog_labels:
                label = 'dog'
            label_list += [label]
            
            if idx % 5 == 4:
                outf.write('\nBlack\t{}'.format(label_list))
                label_list = []
    
        
    with open(infile2) as inf:
        lines = inf.readlines()
        label_list = []
        
        for idx, line in enumerate(lines):
            labels = ' '.join(line.split(' ')[0:-1])
            if ',' in labels:
                label = labels.split(',')[0]
            else:
                label = labels
            if label in monkey_labels:
                label = 'monkey'
            if label in dog_labels:
                label = 'dog'
            label_list += [label]
            
            if idx % 5 == 4:
                outf.write('\nWhite\t{}'.format(label_list))
                label_list = []