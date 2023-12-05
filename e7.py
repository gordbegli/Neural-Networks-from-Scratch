#categorical cross entropy is a loss function used in classification nets.
#custom loss functions might be needed to solve harder problems, but this one seems to work well for classification
import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

#negative sum of targets*ln(outputs)
categorical_cross_entropy_loss = -(math.log(softmax_output[0])*target_output[0] + math.log(softmax_output[1])*target_output[1]+ math.log(softmax_output[2])*target_output[2])
print(categorical_cross_entropy_loss)

#categorical cross entropy simplifies to negative log of predicted value
print(math.log(0.7) * -1)
print(math.log(0.5) * -1)#lower confidence in right answer --> higher loss
