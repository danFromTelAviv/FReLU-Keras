"# FReLU-Keras" 
I slightly modified the PReLU code to make it a FReLU. I have tested it in keras - tensorflow backend ( 1.8 ). It works quite nicely to stabalize the learning. 

USE ( toy example ) : 
input_layer = Input( shape=(None, None, num_channels))
conv_2d = Conv2(num_filters, filter_lens, activation=None)(input_layer)
conv_2d_frelu = FReLU(shared_axes=[1,2])(conv_2d)
conv_2d_BN = BatchNormalization()(conv_2d_frelu)

conv_2d = Conv2(num_filters, filter_lens, activation=None)(conv_2d_BN)
conv_2d_frelu = FReLU(shared_axes=[1,2])(conv_2d)
conv_2d_BN = BatchNormalization()(conv_2d_frelu)

output = Conv2(num_classes, (1,1), activation="softmax")(conv_2d_BN)

model = Model(input_layer, output)
model.compile()



