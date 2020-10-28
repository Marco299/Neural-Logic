:- module(abduction,[
  dig/1,
  add/3
	]).

:- ensure_loaded(library(lists)).

dig(X) :-
  member(X, [0,1,2,3,4,5,6,7,8,9]).

add(X,Y,Z) :-
  dig(X),
  dig(Y),
  Z is X+Y.
