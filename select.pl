#!/usr/bin/perl

$milen=$ARGV[1];

open(FILE, $ARGV[0]);
while(<FILE>){
    chomp;
    @ary = split(/ /, $_);
    if ($ary[1] >= $minlen){
	$tag{$ary[0]} = 1;
    }
}
close(FILE);

while(<STDIN>){
    $line=$_;
    chomp;
    @ary = split(/ /, $_);
    if ($tag{$ary[0]} ne ""){
	print $line;
    }
}
