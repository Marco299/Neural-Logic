:- module(abduction,[
  generate_abductions/2
	]).

:- ensure_loaded(library(lists)).

dig(X) :-
  member(X, [0,1,2,3,4,5,6,7,8,9]).

add(X,Y,Z) :-
  dig(X),
  dig(Y),
  Z is X+Y.

generate_abductions(SumList, AbductionList) :-
    generate_abductions(SumList, 0, _, [], AbductionList).

generate_abductions([], _, _, List, List).
generate_abductions([H|T], IndexIn, IndexOut, ListIn, ListOut) :-
    findall([X,Y], add(X,Y,H), Abductions),
    add_abduction(Abductions, IndexIn, ListIn, ListPart),
    IndexPart is IndexIn+1,
    generate_abductions(T, IndexPart, IndexOut, ListPart, ListOut).

add_abduction(Abductions, SumIndex, [], [[AddendsIndexes, Abductions]]) :-
    generate_indexes(SumIndex, AddendsIndexes),
    !.

add_abduction(Abductions, SumIndex, [[Indexes,Abductions1]|T], UpdatedList) :-
    length(Abductions, AbductionsLen),
    length(Abductions1, Abductions1Len),
    Abductions1Len >= AbductionsLen,
    !,
    generate_indexes(SumIndex, AddendsIndexes),
    PartList = [[AddendsIndexes, Abductions], [Indexes,Abductions1]],
    append(PartList, T, UpdatedList).

add_abduction(Abductions, SumIndex, [[Indexes,Abductions1]|T], UpdatedList) :-
    add_abduction(Abductions, SumIndex, T, PartList),
    append([[Indexes,Abductions1]], PartList, UpdatedList).

generate_indexes(SumIndex, AddendsIndexes) :-
    FirstAddendIndex is SumIndex*2,
    SecondAddendIndex is FirstAddendIndex+1,
    AddendsIndexes = [FirstAddendIndex, SecondAddendIndex].
