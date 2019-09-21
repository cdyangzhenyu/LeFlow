python freeze_graph.py \
--input_meta_graph=./Model/model.ckpt.meta \
--input_checkpoint=./Model/model.ckpt \
--output_graph=./Model/model.pb \
--output_node_name=output \
--input_binary=True

