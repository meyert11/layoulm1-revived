# LayoutLMv1-revived
**Multimodal (text + layout/format + image) pre-training for document AI**

- April 17th, 2021: [LayoutXLM](https://arxiv.org/abs/2104.08836) extends the LayoutLM/LayoutLMv2 into multilingual support! In addition, we also introduce XFUN, a multilingual form understanding benchmark including forms with human labeled key-value pairs in 7 languages (Chinese, Japanese, Spanish, French, Italian, German, Portuguese).
- December 29th, 2020: [LayoutLMv2](https://arxiv.org/abs/2012.14740) is coming with the new SOTA on a wide varierty of document AI tasks, including [DocVQA](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=1) and [SROIE](https://rrc.cvc.uab.es/?ch=13&com=evaluation&task=3) leaderboard.


## Introduction

LayoutLM model is a multimodal transformer to process documents. Although it was introduced in 2020, it still retains high performance for most document classification and personal information extraction. Version 1 was introduced with a permissive license, however versions 2+ now have a non-commercial element, which blocks use in businesses. Microsoft has deprecated v1. The intent of this project is to revive version 1, and to continue development with the permissive Apache 2.0 license. This weekend warrior project is a labor of love, so forgive long pauses in between releases <3

## Pre-trained Model

LayoutLM was pretrained on IIT-CDIP Test Collection 1.0\* dataset with two settings. 

* LayoutLM-Base, Uncased (11M documents, 2 epochs): 12-layer, 768-hidden, 12-heads, 113M parameters || models/base || [OneDrive](https://1drv.ms/u/s!ApPZx_TWwibInS3JD3sZlPpQVZ2b?e=bbTfmM) | [Google Drive](https://drive.google.com/open?id=1Htp3vq8y2VRoTAwpHbwKM0lzZ2ByB8xM) 
* LayoutLM-Large, Uncased || models/large || (11M documents, 2 epochs): 24-layer, 1024-hidden, 16-heads, 343M parameters || [OneDrive](https://1drv.ms/u/s!ApPZx_TWwibInSy2nj7YabBsTWNa?e=p4LQo1) | [Google Drive](https://drive.google.com/open?id=1tatUuWVuNUxsP02smZCbB5NspyGo7g2g)

\*As some downstream datasets are the subsets of IIT-CDIP, we have carefully excluded the overlap portion from the pre-training data.

## Fine-tuning Example

Setup environment as follows:

~~~bash
conda create -n layoutlm_revived python=3.11
conda activate layoutlm_revived
pip install -r requirements.txt

~~~

### Sequence Labeling Task


I'm starting this off by fine-tuning an example for sequence labeling tasks. You can run this example dataset in /data/funsd or on [FUNSD](https://guillaumejaume.github.io/FUNSD/), a dataset for document understanding tasks.

First, we need to preprocess the JSON file into txt. You can run the preprocessing scripts `funsd.py` in the `src` directory. For more options, please refer to the arguments, the preprocessing is already included in the included example notebooks.

~~~bash
Example_large.ipynb or Example_base.ipynb
~~~

I ran both on RTX 3080's with 10 GB of VRAM, so it can definitely fit on most GPUs.

## Citation

If you find LayoutLM useful in your research, please cite the following paper:

``` latex
@misc{xu2019layoutlm,
    title={LayoutLM: Pre-training of Text and Layout for Document Image Understanding},
    author={Yiheng Xu and Minghao Li and Lei Cui and Shaohan Huang and Furu Wei and Ming Zhou},
    year={2019},
    eprint={1912.13318},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## License

Apache 2.0 - see file

### Contact Information

For help or issues using LayoutLMv1-revived, please submit a GitHub issue.

For other communications related to to this project, please contact Travis Meyer (meyert11 at gmail.com).
