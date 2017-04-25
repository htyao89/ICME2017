#!/usr/bin/env sh

../../../build/tools/extract_features.bin ../models/reid_20161208_iter_50000.caffemodel  deploy_gallery.prototxt  feat  reid_gallery_feat 52534  leveldb GPU 1
rm -rf reid_gallery_feat

../../../build/tools/extract_features.bin ../models/reid_20161208_iter_50000.caffemodel  deploy_query.prototxt  feat  reid_query_feat 483  leveldb GPU 1
rm -rf reid_query_feat

../../../build/tools/extract_features.bin  ../models/reid_20161208_iter_50000.caffemodel  deploy_train.prototxt  feat  reid_train_feat 1711  leveldb GPU 1
rm -rf reid_train_feat

