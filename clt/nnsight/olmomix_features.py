
from datasets import Features, Value, Sequence

# manually defining the schema for each of the features in the olmo-mix-datset as a workaround
# Works
wiki_features = Features({
    'text': Value('string'),
    'added': Value('string'),
    'created': Value('string'),
    'id': Value('string'),
    'metadata': Features({
        'length': Value('int64'),
        'provenance': Value('string'),
        'revid': Value('string'),
        'url': Value('string')
    }),
    'source': Value('string'),
    'version': Value('string')
})

# Works
pes2o_features = Features({
    'text': Value('string'),
    'added': Value('string'),
    'created': Value('string'),
    'attributes': Value('string'),
    'doc': Value('string'),
    'id': Value('string'),
    'metadata': Features({
        "abstract": Value('string'),
        "abstract_count": Value('int64'),
        "abstract_language": Value("string"),
        "abstract_perplexity": Value("float64"),
        "extfieldsofstudy": Sequence(Value("string")),
        "provenance": Value("string"),
        "s2fieldsofstudy": Sequence(Value("string")),
        "sha1": Value("string"),
        "sources": Sequence(Value("string")),
        "title": Value("string"),
        "title_count": Value("int64"),
        "title_language": Value("string"),
        "title_perplexity": Value("float64"),
        "top_frequencies": [{
            "count": Value("int64"),
            "token": Value("string")
        }],
        "year": Value("int64")
    }),
    'source': Value('string'),
    'version': Value('string')
})

# works
algebraicstack_features = Features({
    'text': Value('string'),
    'added': Value('string'),
    'created': Value('string'),
    'attributes': Features({
        "paloma_paragraphs": Sequence(Sequence(Value('int64'))),
        "paloma_documents": Value('string')
    }),
    "doc": Features({
        "arxiv_id": Value("string"),
        "language": Value("string"),
        "timestamp": Value("int64"),
        "url": Value("string"),
        "yymm": Value("string")
    }),
    'id': Value('string'),
    'metadata': Features({
        'provenance': Value('string'),
    }),
    'source': Value('string'),
    'version': Value('string')
})

# works
arxiv_features = Features({
    'text': Value('string'),
    'added': Value('string'),
    'created': Value('string'),
    'attributes': Features({
        "paloma_paragraphs": Sequence(Sequence(Value('int64'))),
        "paloma_documents": Value('string')
    }),
    "doc": Features({
        "arxiv_id": Value("string"),
        "language": Value("string"),
        "timestamp": Value("int64"),
        "url": Value("string"),
        "yymm": Value("string")
    }),
    'id': Value('string'),
    'metadata': Features({
        'provenance': Value('string'),
    }),
    'source': Value('string'),
    'version': Value('string')
})

#TODO - datasetviewer not available for this split so I'll need to download and manually analyze
dclm_features = Features({
    'text': Value('string'),
    'added': Value('string'),
    'created': Value('string'),
    'id': Value('string'),
    'metadata': Features({
        'length': Value('int64'),
        'provenance': Value('string'),
        'revid': Value('string'),
        'url': Value('string')
    }),
    'source': Value('string'),
    'version': Value('string')
})

# works
openwebmath_features = Features({
    'text': Value('string'),
    'added': Value('string'),
    'created': Value('string'),
    'attributes': Features({
        "paloma_paragraphs": Sequence(Value("string"))
    }),
    "doc": Features({
        "config": Features({
            "boilerplate_config": Features({
                "absolute_threshold": Value("int64"),
                "enable": Value("bool"),
                "end_threshold": Value("int64"),
                "ratio_threshold": Value("float64")
            }),
            "extract_latex": Value("bool"),
            "markdown_code": Value("bool"),
            "markdown_headings": Value("bool"),
            "remove_buttons": Value("bool"),
            "remove_chinese": Value("bool"),
            "remove_edit_buttons": Value("bool"),
            "remove_image_figures": Value("bool"),
            "remove_link_clusters": Value("bool"),
            "table_config": Features({
                "format": Value("string"),
                "min_cols": Value("int64"),
                "min_rows": Value("int64")
            })
        }),
        "date": Value("int64"),
        "extraction_info": Features({
            "/images/math/codecogs": Value("int64"),
            "align": Value("int64"),
            "codecogs_latex": Value("int64"),
            "equation": Value("int64"),
            "found_math": Value("bool"),
            "img_math": Value("int64"),
            "katex": Value("int64"),
            "math-container": Value("int64"),
            "math_alttext": Value("int64"),
            "math_annotations": Value("int64"),
            "math_score": Value("float64"),
            "mathjax_asciimath": Value("int64"),
            "mathjax_display_tex": Value("int64"),
            "mathjax_inline_tex": Value("int64"),
            "mathjax_tag": Value("int64"),
            "mathml": Value("int64"),
            "mathtex.cgi": Value("int64"),
            "mimetex.cgi": Value("int64"),
            "perplexity": Value("float64"),
            "script_math_asciimath": Value("int64"),
            "script_math_tex": Value("int64"),
            "texerror": Value("int64"),
            "wp-katex-eq": Value("int64"),
            "wp_latex": Value("int64"),
            "x-ck12": Value("int64")
        }),
        "url": Value("string"),
        "warc_path": Value("string")
    }),
    'id': Value('string'),
    'metadata': Features({
        'provenance': Value('string'),
    }),
    'source': Value('string'),
    'version': Value('string')
})

# works
starcoder_features = Features({
    'text': Value('string'),
    'added': Value('string'),
    'created': Value('string'),
    'attributes': Features({
        "top_20_tokens__top_20_tokens__p1": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p2": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p3": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p4": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p5": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p6": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p7": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p8": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p9": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p10": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p11": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p12": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p13": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p14": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p15": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p16": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p17": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p18": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p19": Sequence(Sequence(Value("float64"))),
        "top_20_tokens__top_20_tokens__p20": Sequence(Sequence(Value("float64"))),
        "whitespace_tokenizer_v1__whitespace_tokenizer_v1__length": Sequence(Sequence(Value("float64"))),
        "tokenizer_repetitions_v2r2__tokenizer_repetitions_v2r2__doc_frac_repetition": Sequence(Sequence(Value("float64"))),
        "tokenizer_repetitions_v2r2__tokenizer_repetitions_v2r2__doc_max_length_repetition": Sequence(Sequence(Value("float64"))),
        "tokenizer_repetitions_v2r2__tokenizer_repetitions_v2r2__doc_max_score_repetition": Sequence(Sequence(Value("float64"))),
        "tokenizer_repetitions_v2r2__tokenizer_repetitions_v2r2__repetition": Sequence(Sequence(Value("float64")))
    }),
    "doc": Value('string'),
    'id': Value('string'),
    'metadata': Features({
        "extension": Value("string"),
        "max_stars_count": Value("string"),  # or "int64" if you convert it numerically
        "max_stars_repo_name": Value("string"),
        "max_stars_repo_path": Value("string"),
        "provenance": Value("string")
    }),
    'source': Value('string'),
    'version': Value('string')
})


hf_data_files = ['data/wiki/*',
             'data/pes2o/*',
             'data/algebraic-stack/**',
            #  'data/arxiv/**',
            #  'data/dclm/**',
            #  'data/openwebmath/**',
            #  'data/starcoder/**'
             ]

# hf_data_files = ['data/wiki/*',
#              'data/pes2o/*',
#              'data/algebraic-stack/**',
#              'data/arxiv/**',
#              'data/dclm/**',
#              'data/openwebmath/**',
#              'data/starcoder/**']