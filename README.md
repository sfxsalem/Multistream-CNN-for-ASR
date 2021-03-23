
# Multistream CNN for ASR

Implementation of a ["Multistream CNN for Robust Acoustic Modeling"](https://arxiv.org/abs/2005.10470) using ["Pytorch-Kaldi"](https://github.com/mravanelli/pytorch-kaldi) and Librispeech Dataset

* For a full description of the project, please read the documentation included

in the repository:
```https://github.com/sfxsalem/Multistream-CNN-for-ASR/tree/master/docs```

Getting Started
-------------
This research project aims to implement the new multistream CNN architecture for robust acoustic modeling in speech recognition tasks. The proposed architecture is decribed throughly in ["Multistream CNN for Robust Acoustic Modeling"](https://arxiv.org/abs/2005.10470). The goal of this work is to harness both the efficiceny of Kaldi and the flexibility of Pytorch to better represent acoustic events without having to drastically increase the model's complexity.

["Pytorch-Kaldi"](https://github.com/mravanelli/pytorch-kaldi) embeds several useful features for developing modern speech recognizers. For instance, the code is specifically designed to allow developpers to plug-in user-defined models and to combine those thus enabling the use of complex neural architectures. The toolkit is publicly-released along with rich documentation and tutorials .

Prerequisites
-------------
To get started with this project, first follow the instructions below:


 1. Start by installing ["Kaldi"](http://kaldi-asr.org/). Add the path to Kaldi binaries into ~/.bashrc and source it.
	Use ```copy-feats``` or ```hmm-info``` to test the installation. If no errors appear then everything is fine.

 2. Install ["PyTorch"](http://pytorch.org/) and the equivalent ["CUDA libraries"](https://developer.nvidia.com/cuda-downloads).
 3. Clone the PyTorch-Kaldi repository and install the required packages:
	```
	git clone https://github.com/mravanelli/pytorch-kaldi
	cd pytorch-kaldi
	pip install -r requirements.txt
	```
Librispeech Dataset Pre-processing
-------------
The following steps provide a short tutorial on how to prepare the Librispeech dataset to work with Pytorch-Kaldi. Note that this is provided for the 100 hours subset, with few changes this could be extended to the full 960 hours dataset:

 1. Run the Kaldi recipe for librispeech until Stage 13 (included)
 2. Decode test and dev datasets
	```
	utils/mkgraph.sh data/lang_test_tgsmall \
						exp/tri4b exp/tri4b/graph_tgsmall
	for test in test_clean dev_clean; do
		steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
								exp/tri4b/graph_tgsmall data/$test exp/tri4b/decode_tgsmall_$test
		steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
								data/$test exp/tri4b/decode_{tgsmall,tgmed}_$test
		steps/lmrescore_const_arpa.sh \
			--cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
			data/$test exp/tri4b/decode_{tgsmall,tglarge}_$test
		steps/lmrescore_const_arpa.sh \
			--cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
			data/$test exp/tri4b/decode_{tgsmall,fglarge}_$test
	done
	```
 3. Copy ```exp/tri4b/trans.*``` files into ```exp/tri4b/decode_tgsmall_train_clean_100/```
	```
	mkdir exp/tri4b/decode_tgsmall_train_clean_100 && cp exp/tri4b/trans.* exp/tri4b/decode_tgsmall_train_clean_100/
	```
 4. Compute the fmllr features by running the following script.
	```
	. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
	. ./path.sh ## Source the tools/utils (import the queue.pl)

	gmmdir=exp/tri4b

	for chunk in train_clean_100 dev_clean test_clean; do
		dir=fmllr/$chunk
		steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
			--transform-dir $gmmdir/decode_tgsmall_$chunk \
				$dir data/$chunk $gmmdir $dir/log $dir/data || exit 1
	 
		compute-cmvn-stats --spk2utt=ark:data/$chunk/spk2utt scp:fmllr/$chunk/feats.scp ark:$dir/data/cmvn_speaker.ark
	done
	```
 5. compute Alignements using:
	```
	# aligments on dev_clean and test_clean
	steps/align_fmllr.sh --nj 30 data/train_clean_100 data/lang exp/tri4b exp/tri4b_ali_clean_100
	steps/align_fmllr.sh --nj 10 data/dev_clean data/lang exp/tri4b exp/tri4b_ali_dev_clean_100
	steps/align_fmllr.sh --nj 10 data/test_clean data/lang exp/tri4b exp/tri4b_ali_test_clean_100
	```
Implementation of Multistream CNN
-------------

 1. Clone this repository and install the required packages:
	```
	git clone https://github.com/sfxsalem/Multistream-CNN-for-ASR.git
	cd Multistream-CNN-for-ASR
	pip install -r requirements.txt
	```
2. Copy the files into the pytorch-kaldi directory :
	```
	cp -r * ~/pytorch-kaldi
	```
 3. run the experiments with the following command:
	```
	cd ~/pytorch-kaldi
	python run_exp.py cfg/Librispeech_baselines/libri_MBS_fmllr.cfg
	```