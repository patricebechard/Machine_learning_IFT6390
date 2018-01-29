import json

architectures = []

#n_filters_list = [[10], [50], [100], [10, 50], [50, 100], [10, 50, 100]]
#hid_layers_dim_list = [[10], [50], [100], [50, 10], [100, 50], [100, 50, 10]]
activations_dnn_list = ["sigmoid", "relu", "softmax"]

n_filters_list = [[10, 50, 100, 200]] #[100],[50,100],[50,100,200], 
hid_layers_dim_list = [[50]]

for n_filters in n_filters_list:
		for hid_layers_dim in hid_layers_dim_list:
				config = {}
				
				config["n_filters"] = n_filters
				config["activations_cnn"] = ["relu" for i in range(len(n_filters))]
				config["hid_layers_dim"] = hid_layers_dim
				if hid_layers_dim is None:
					config["activations_dnn"] = activations_dnn_list[0]
				elif len(hid_layers_dim) == 1:
					config["activations_dnn"] = [activations_dnn_list[1], activations_dnn_list[2]]
				elif len(hid_layers_dim) == 2:
					config["activations_dnn"] = [activations_dnn_list[1], activations_dnn_list[1], activations_dnn_list[2]]
				elif len(hid_layers_dim) == 3:
					config["activations_dnn"] = [activations_dnn_list[1], activations_dnn_list[1], activations_dnn_list[1], activations_dnn_list[2]]
				
				architectures.append(config)
				
with open('architectures_cnn.json', 'w') as outfile:  
    json.dump(architectures, outfile)