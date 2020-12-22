:- module(abduction,[
  generate_abductions/3
	]).

:- ensure_loaded(library(lists)).

dig(X) :-
  member(X, [0,1,2,3,4,5,6,7,8,9]).

add(Addends,AddendsNum,Sum) :-
    add(Addends, 0, _, AddendsNum, 0, Sum).

add([], AddendsNum, _, AddendsNum, Sum1, Sum) :-
    !,
    Sum is Sum1.

add([H|T], AddendsNumIn, AddendsNumOut, AddendsNum, SumIn, SumOut) :-
    dig(H),
    AddendsNumPart is AddendsNumIn + 1,
    SumPart is SumIn + H,
    add(T, AddendsNumPart, AddendsNumOut, AddendsNum, SumPart, SumOut).

generate_abductions(SumList, AddendsNum, AbductionList) :-
    generate_abductions(SumList, AddendsNum, 0, _, [], AbductionList).

generate_abductions([], _, _, _, List, List).
generate_abductions([H|T], AddendsNum, IndexIn, IndexOut, ListIn, ListOut) :-
    findall(L, add(L, AddendsNum, H), Abductions),
    add_abduction(Abductions, AddendsNum, IndexIn, ListIn, ListPart),
    IndexPart is IndexIn+1,
    generate_abductions(T, AddendsNum, IndexPart, IndexOut, ListPart, ListOut).

add_abduction(Abductions, AddendsNum, SumIndex, [], [[AddendsIndexes, Abductions]]) :-
    generate_indexes(SumIndex, AddendsNum, AddendsIndexes),
    !.

add_abduction(Abductions, AddendsNum, SumIndex, [[Indexes,Abductions1]|T], UpdatedList) :-
    length(Abductions, AbductionsLen),
    length(Abductions1, Abductions1Len),
    Abductions1Len >= AbductionsLen,
    !,
    generate_indexes(SumIndex, AddendsNum, AddendsIndexes),
    PartList = [[AddendsIndexes, Abductions], [Indexes,Abductions1]],
    append(PartList, T, UpdatedList).

add_abduction(Abductions, AddendsNum, SumIndex, [[Indexes,Abductions1]|T], UpdatedList) :-
    add_abduction(Abductions, AddendsNum, SumIndex, T, PartList),
    append([[Indexes,Abductions1]], PartList, UpdatedList).

generate_indexes(SumIndex, AddendsNum, AddendsIndexes) :-
    FirstIndex is SumIndex*AddendsNum,
    generate_indexes(FirstIndex, AddendsNum, [FirstIndex], AddendsIndexesRev),
    reverse(AddendsIndexesRev, AddendsIndexes).

generate_indexes(_, AddendsNum, AddendsIndexes, AddendsIndexes) :-
    length(AddendsIndexes, AddendsIndexesLen),
    AddendsNum == AddendsIndexesLen,
    !.

generate_indexes(Index, AddendsNum, AddendsIndexesIn, AddendsIndexesOut) :-
    NewIndex is Index + 1,
    append([NewIndex], AddendsIndexesIn, AddendsIndexesPart),
    generate_indexes(NewIndex, AddendsNum, AddendsIndexesPart, AddendsIndexesOut).

