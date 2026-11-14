# K2 text schema validation

On 2026-11-14, two live rows from each of the 24 pretraining/SFT subsets in
`sources.K2` passed their registered formatter and matched `row["text"]` exactly.
No corpus text is committed; offline test fixtures are invented examples.
`validation.json` records repository revisions reported during inspection, field
names and subset coverage. Dataset viewer rows were requested with offset=0 and
length=2. The viewer was unavailable for math-dialogue and 3efforts-pretrain;
those two samples were read directly from revision-pinned source Parquet files.

Every observed subset has preformatted text. TxT360's extra metadata was excluded;
chat markers, reasoning and code indentation were preserved, not re-templated.
This is a schema/formatting check, not a corpus quality or language-model result.
It does not establish that every row in the corpus is valid. Malformed-row tests
ensure missing/blank/non-string text cannot turn metadata into a training document.
The separate guru-RL task/reward schema is outside this registry.
