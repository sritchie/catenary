(* ::Package:: *)

(* ::Chapter:: *)
(*Generating the Loop Equations*)


BeginPackage["LoopEquations`"];


(* ::Section:: *)
(*Words, Symmetries, Bracelets and Necklaces*)


(* ::Text:: *)
(*The general idea is that we've got some partition of the set of permutations of matrices; we'd like to map every element of some partition down to a single "canonical representation" of the partition, then simplify the loop equations down further.*)
(**)
(*The trace is cyclically symmetric, and symmetric under reversal.This means that every element in a bracelet has the same trace.*)
(**)
(*Therefore we can get some simplification out of the gate by mapping each sequence of matrices down to some "canonical element" of the bracelet. As long as the symbols we use have an ordering, we can simply expand some element out to its bracelet, sort and take the first element.*)


(* ::Subsection:: *)
(*Symmetry Functions*)


(* ::Text:: *)
(*Here are some symmetries we're interested in.*)


(* ::Input::Initialization:: *)
id::usage = "Returns a singleton list of the input. An ID symmetry, if you will.";

rotorSymmetry::usage = "Takes a 'rotor', ie, the elements of a cyclic rotor, and returns a symmetry fn that generates all bike-lock symmetries.";

paired::usage = "Returns a symmetry generator that swaps reflected elements.";

cycles::usage = "Generates all cyclic rotations of the input list.";

reflection::usage = "Generates a reflection symmetry, reversing the args.";

Begin["`Private`"];

id[xs_] := {xs};

rotorSymmetry[] = id;
rotorSymmetry[{x_}] := id;
rotorSymmetry[rotor_] := Module[
{f, spin, m = AssociationThread[rotor, RotateLeft[rotor]]},
spin[xs_] := Lookup[m, #, #]& /@ xs;
f[xs_] := NestList[spin, xs, Length[rotor] - 1];
f];

cycles[{}] := {{}};
cycles[xs_] := NestList[RotateLeft,xs,Length[xs]-1];

reflection[xs_] := {xs, Reverse[xs]}

End[];


(* ::Text:: *)
(*And then the ability to compose them:*)


(* ::Input::Initialization:: *)
compose::usage = "Composition for symmetries. Returns a new symmetry generator.";

canonical::usage = "Returns a function that returns the representative element after taking all symmetries into account.";

Begin["`Private`"];

compose[] := id;
compose[f_] := f
compose[f_, g_] := DeleteDuplicates[Join @@ g /@ f[#]] &;
compose[f_, g_, rest__] := compose[compose[f, g], rest]

canonical[symmetryFns__] := Module[
{f, symFn = compose @@ {symmetryFns}},
f[xs_] := First @ Sort @ symFn[xs];
f[xs__] := DeleteDuplicates[f /@ {xs}];
f
];

End[];


(* ::Text:: *)
(*Here are some common symmetries that we're interested in:*)


(* ::Input::Initialization:: *)
necklace::usage = "Alias for cycles; fn that returns the full necklace that this element belongs to.";

bracelet::usage = "fn that returns the full bracelet that this element belongs to.";

z2Symmetry::usage = "Returns a pair of the original input element, plus the element with all 'A' and 'B' flipped.";

braceletZ2::usage = "Returns the bracelet, and the bracelet of the z2-symmetry of the input.";

Begin["`Private`"];

necklace = cycles;
bracelet = compose[necklace, reflection];
z2Symmetry = rotorSymmetry[{"A", "B"}];
braceletZ2 = compose[necklace, reflection, z2Symmetry];

End[];


(* ::Subsection:: *)
(*Generators*)


(* ::Text:: *)
(*Now, an actual generator that we can apply the symmetries to. You can pass the output of "words" into the function returned by "canonical" to get a list of distinct elements!*)


(* ::Input::Initialization:: *)
words::usage = "Generates all words of length k from an alphabet of n items.";

Begin["`Private`"];

words[alphabetSize_Integer, wordLength_Integer] :=Module[
{nLetters = ToUpperCase[Alphabet[]][[;; alphabetSize]]},
Tuples[nLetters,wordLength]
];
words[alphabetSize_Integer, wordLength_List] :=
Join @@ (words[alphabetSize, #]&  /@ wordLength);

End[];


(* ::Section:: *)
(*Loop Equations from a Polynomial Model*)


(* ::Subsection:: *)
(*Model*)


(* ::Text:: *)
(*Some code for describing a model itself.*)


(* ::Input::Initialization:: *)
model::usage =
 "Takes a series of {coefficient, model term} pairs, plus any number of symmetries,
  and returns the non-quadratic terms of the model with all symmetries expanded.";

matrixCount::usage = "returns the number of matrices in the model.";

interactionVariables::usage = "Returns the non-quadratic terms in the model.";

Begin["`Private`"];

normalizeTerm[matrixTerm_List] := ToString /@ matrixTerm;
normalizeTerm[matrixTerm_Symbol] := ToString[matrixTerm];
normalizeTerm[matrixTerm_] := Module[
{factors = Table @@ #&  /@ Drop[FactorList[matrixTerm], 1]},
normalizeTerm[Join @@ factors]
];
normalizeTerm::usage = "Normalizes polynomial terms in the model.";

model[terms_] := MapAt[normalizeTerm, terms, {All, 2}];
model[terms_, symmetryFns__] := Module[
{f, symFn = compose @@ {symmetryFns}},
f[{coef_, term_}] := {coef, #}& /@ symFn[term];
Join @@ f /@ model[terms]
];

matrixCount[model_] := CountDistinct[Join @@ (#[[2]]& /@ model)];

quadraticQ[{1, {l_, r_}}] := l === r;
quadraticQ[term_] := False;

quadraticVariables[model_] :=
DeleteDuplicates[#[[2, 1]]& /@ Select[model, quadraticQ]];
quadraticVariables::usage = "Returns the distinct set of quadratic variables in the model.";

interactionVariables[model_] :=
Select[model, !quadraticQ[#]&];

End[];


(* ::Subsection:: *)
(*Quadratic Terms*)


(* ::Text:: *)
(*The quadratic terms. Time to chop up some blobs. Here are some utilities for generating paths through the blob.*)


(* ::Input::Initialization:: *)
quadraticTerms::usage = "Quadratic terms, which come from splitting up the planar graph in various ways.";

quadraticTermFn::usage = "Curried version of quadraticTerms.";

Begin["`Private`"];

otherLocations[blob_, i_Integer] :=
Module[
{positions = Position[blob, blob[[i]]], other},
Flatten[DeleteCases[positions, {i}]]
];
otherLocations::usage =
 "Takes a list of terms and a position, and returns all OTHER places that the term at position i appears.";

routes[i_Integer, exits_] := Sort@{i,#}&/@ exits;
routes::usage = "Returns a sorted list of pairs of the form {entrance_index, exit_index}
through a blob, from i to each of the items in 'exits'.
";

sliceBlob[blob_, {start_, end_}] := sliceBlob[blob, start, end];
sliceBlob[blob_, start_, end_] :=
 Module[{l, r},
{l, r} = TakeDrop[blob, {start, end}];
{Delete[l, {{-1}, {1}}], r}
];
sliceBlob::usage =
"Returns a pair with both halves of the blob that remain after slicing through a path from start to end.";


(* ::Text:: *)
(*And here's the code that actually generates the terms.*)


(* ::Input::Initialization:: *)
quadraticTerms[toCorrelator_, blob_, i_Integer]:=
Module[
{blobRoutes, product, slice, exits=otherLocations[blob, i]},

slice[route_] := sliceBlob[blob, route];
product[{l_, r_}] := toCorrelator[l] * toCorrelator[r];

If[Length[exits] ==0,
0,
blobRoutes = routes[i, exits];
Total[(product @* slice) /@ blobRoutes]
]
];
quadraticTerms::usage = "Quadratic terms, which come from splitting up the planar graph in various ways.";

quadraticTermFn[model_, toCorrelator_] := Module[
{quads = quadraticVariables[model], f},
f[blob_, i_Integer] := If[MemberQ[quads, blob[[i]]],
quadraticTerms[toCorrelator, blob, i],
0];
f
];
quadraticTermFn::usage = "Curried version of quadraticTerms.";

End[];


(* ::Subsection:: *)
(*Interaction Terms*)


(* ::Text:: *)
(*Now we can write the terms that result from interactions described by the model.*)


(* ::Input::Initialization:: *)
termReplacements::usage = "Generates an association of element to the other spokes on its wheel.";

termReplacementFn::usage =
 "Similar to termReplacements, but instead of returning
  an association, returns a total function that can handle defaults
  missing from the association. (If an item is missing, the function generates
  an empty list, meaning, generate no replacements.

  Returns a function from an element to the items that it could expand to. ";

interaction::usage =
"Take a coefficient and a polynomial term, and returns a function of (blob, i) that produces all possible interactions.";

interactionTermFn::usage =
 "Takes pairs of model terms and a toCorrelator fn, and returns a function of
   (blob, i) that generates all of the model's interaction terms.";

Begin["`Private`"];

termReplacements[{}] = <||>;
termReplacements[term_] := Module[{distinctTails},
distinctTails[xs_] := DeleteDuplicates @ Map[Rest, xs];
 GroupBy[cycles[term], First, distinctTails]
];

termReplacementFn[term_] := Lookup[termReplacements[term], #, {}] &;

extract[blob_, interaction_, i_Integer] :=  Module[{l, r},
{l, r} = TakeDrop[blob, i];
Join [Drop[l, -1], interaction,  r]
];
extract::usage =
"extracts the i'th term of the blob xs and replaces it with the supplied interaction term.";

interaction[term_] :=  Module[{mFn, f},
mFn = termReplacementFn[term];
f[{}, i_Integer] := {{}};
f[blob_, i_Integer] :=  Module[{v = blob[[i]]},
extract[blob, #, i]& /@ mFn[v]
];
f
];

termExpander[{coef_, term_}, toCorrelator_] := Module[
{ret, f = interaction[term]},
ret[blob_, i_]:= (
coef * Total[toCorrelator /@ f[blob, i]]
);
ret
];
termExpander::usage = "Returns a function of blob, i that expands out the summed correlators with the coefficient multiplied.";

interactionTermFn[model_, toCorrelator_] := Module[
{pairs = interactionVariables[model], ret, fs},
fs = Map[termExpander[#, toCorrelator]&, pairs];
ret[blob_, i_Integer] := Simplify[Total[#[blob, i]& /@ fs]];
ret
];

End[];


(* ::Subsection:: *)
(*Correlator Functions*)


(* ::Text:: *)
(*Here' s some code to describe correlators in a way that captures their symmetries, and turns them into proper subscripted variable entries.*)


(* ::Input::Initialization:: *)
correlator::usage =
"Returns a function that generates a correlator term on the supplied variable e.
  the subscript is the canonical representation, taking into account the
  the supplied symmetries.";

distinctBy::usage = "Filters words down to a list of ONLY words that generate distinct correlators.";

Begin["`Private`"];

formal[e_, {}] = 1;
formal[e_, xs_] := Subscript[e, StringJoin[xs]];
formal::usage =
 "Function that takes a list and returns a subscript of some indexed symbol; for the empty list, returns 1.";

correlator[e_] := formal[e, #] &;
correlator[e_, symmetryFns__] := formal[e, (canonical @@ {symmetryFns})[#]] &;


(* This is cheating a little, since we're baking in the first, sort fetch of the canonical element. *)
distinctBy[toCorrelator_, words_] :=
 Values @ GroupBy[words, toCorrelator, First @* Sort];

End[];


(* ::Subsection:: *)
(*Loop Equations*)


(* ::Text:: *)
(*Now let' s build a few functions that can piece these together.*)


(* ::Input::Initialization:: *)
constraintFn::usage =
 "Returns a function that takes:

- a blob and a position, and generates the constraint from following in the line at index i, or
- JUST a blob; in this case it returns a list of all distinct constraints.";

loopEquations::usage = "Generates the loop equations for words of length k, where k can be a single int or a list of numbers.";

Begin["`Private`"];

constraintFn[model_, toCorrelator_]:= Module[
{quad = quadraticTermFn[model, toCorrelator],
ixn = interactionTermFn[model, toCorrelator],
f},
f[blob_] := DeleteDuplicates[f[blob, #]& /@ Range[Length[blob]]];
f[blob_, i_Integer] := toCorrelator[blob] == ixn[blob,i]+quad[blob,i];
f];

loopEquations[model_, toCorrelator_, k_]:= Module[
{f = constraintFn[model, toCorrelator],
blobs = distinctBy[toCorrelator, words[matrixCount[model], k]]},
Join @@ (f /@ blobs)
];

End[];

EndPackage[];
