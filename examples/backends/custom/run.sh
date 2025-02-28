export LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

export PIPELINE_CONF=$(hostname).yaml
export ENV_CLIENT=quebec
export ENV_DEVICE=intel-a2000

tritonserver --model-repository=/data/repos/viztel-shopassist-prod/triton-model-dir --model-control-mode=explicit --load-model=receipt_tokenizer --load-model=receipt_seg --load-model=barcode_ensemble --load-model=roller_count_process --load-model=size_classify_pre --load-model=receipt_pre --load-model=rpn_movement --load-model=rpn_seg_process --load-model=barcode_det_post --load-model=aruco_decode --load-model=hmm_yolo_pre --load-model=primary-ensemble --load-model=hmm_featurizer --load-model=featuresize_post --load-model=roller_count --load-model=vit --load-model=hmm_featurizer_pre --load-model=parseq_torch --load-model=rpn_post --load-model=bls_barcode --load-model=vit_linear_head --load-model=roller_count_pre --load-model=receipt_process --load-model=receipt_post --load-model=hmm --load-model=fused_ensemble_rpn_pre --load-model=hmm_yolo --load-model=featuresizeclassify --load-model=end_filter --load-model=primary_classifier_post --load-model=projection_logic --load-model=barcode_det --load-model=identity --load-model=parseq_post --load-model=aruco_decode_post --load-model=roller_count-ensemble --load-model=shopassist-ensemble --load-model=rpn_seg --load-model=parseq_receipt --allow-gpu-metrics true --pinned-memory-pool-byte-size 2073741824  --cuda-memory-pool-byte-size 0:2000656000 --model-load-thread-count 1 --exit-timeout-secs 0

