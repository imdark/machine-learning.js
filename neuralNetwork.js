// naieve nural network
// trying to find a function such as or(a,b) = h0 * (a * w00 + b * w10 + w20) + h1 * (a * w01 + b * w11 + w21) + h2 * (a * w02 + b * w12 + w22) , a,b = { 0 or 1 }

function train(trainingSet) {
	var firstRecord = trainingSet[0];

	// number of weights should be the number of features and every record should have the same number of features so the first one will do
	// one more weight for bias
	var weights = [];
	for(var i = 0; i < firstRecord.features.length; i++) {

		var newWeightsArray = [];
		weights.push(newWeightsArray);
		for(var j = 0; j < firstRecord.features.length + 1; j++) {
			newWeightsArray.push(Math.random());
		}
	}

	var hiddenLayerToOutputLayerWeights = [];
	for(var i = 0; i < firstRecord.features.length + 1; i++) {
		hiddenLayerToOutputLayerWeights.push(Math.random());
	}

	for(record of trainingSet) {

		var predictionsForInputLayer = [];
		for(weightsForInputLayer of weights) {
			var predictionForCurrentHiddenLayerNeuron = predictBasedOnWeights(record.features, weightsForInputLayer);
			predictionsForInputLayer.push(predictionForCurrentHiddenLayerNeuron);
		}
		
		var prediction = predictBasedOnWeights(predictionsForInputLayer, hiddenLayerToOutputLayerWeights) > 0 ? 1 : 0;

		// backpropegation: start at the output layer and compare prediction and actual results and use that to adjust weights
		for(predictionIndex in predictionsForInputLayer) {
			hiddenLayerToOutputLayerWeights[predictionIndex] += predictionsForInputLayer[predictionIndex] * (record.result - prediction) * 0.1;
		}

		hiddenLayerToOutputLayerWeights[hiddenLayerToOutputLayerWeights.length -1] += (record.result - prediction) * 0.1;

		// backpropegation: compare the value of the nuron to the predicted value the 
		for(var featureIndexPerHiddenNeuron = 0; featureIndexPerHiddenNeuron < record.features.length; featureIndexPerHiddenNeuron++) {
			var currentNeuronWeights = weights[featureIndexPerHiddenNeuron];
			for(var featureIndex = 0; featureIndex < record.features.length; featureIndex++) {
				currentNeuronWeights[featureIndex] += record.features[featureIndex] * (hiddenLayerToOutputLayerWeights[featureIndexPerHiddenNeuron] - predictionsForInputLayer[featureIndexPerHiddenNeuron]) * 0.1;
			}

			// for the last bais weight
			currentNeuronWeights[currentNeuronWeights.length - 1] += (hiddenLayerToOutputLayerWeights[featureIndexPerHiddenNeuron] - predictionsForInputLayer[featureIndexPerHiddenNeuron]) * 0.1;
		}
	}

	return { inputWeights: weights, outputWeights: hiddenLayerToOutputLayerWeights };
}

function predictBasedOnNetWeights(recordFeatures, allWeights) {
	var weights = allWeights.inputWeights;
	var hiddenLayerToOutputLayerWeights = allWeights.outputWeights;

	var predictionsForInputLayer = [];
	for(weightsForInputLayer of weights) {
		var predictionForCurrentHiddenLayerNeuron = predictBasedOnWeights(record.features, weightsForInputLayer);
		predictionsForInputLayer.push(predictionForCurrentHiddenLayerNeuron);
	}
	
	return predictBasedOnWeights(predictionsForInputLayer, hiddenLayerToOutputLayerWeights) > 0 ? 1 : 0;
}

function predictBasedOnWeights(recordFeatures, weights) {
	
	// add 1 for bias
	var inputFeatures = recordFeatures.concat(1);

	var numberOfTestFeatures = inputFeatures.length;
	var predictedResult = 0;

	for(featureIndex in inputFeatures) {
		predictedResult += inputFeatures[featureIndex] * weights[featureIndex];
	}
	
	return predictedResult;
}

var xorGate = [
	{features: [1,1],result: 0},
	{features: [1,0],result: 1},
	{features: [0,1],result: 1},
	{features: [0,0],result: 0}
];

var orGate = [
	{features: [1,1],result: 1},
	{features: [1,0],result: 1},
	{features: [0,1],result: 1},
	{features: [0,0],result: 0}
];

var andGate = [
	{features: [1,1],result: 1},
	{features: [1,0],result: 0},
	{features: [0,1],result: 0},
	{features: [0,0],result: 0}
];

// sampling helps improve naievness
function sample(array, size) {
	var result = Array(size);
	for(var i = 0; i < size; i++) {
		result[i] = array[Math.floor(Math.random() * array.length)];
	}
	return result;
}

predictBasedOnNetWeights([1,1], train(sample(orGate,1000))) // => 1
predictBasedOnNetWeights([1,0], train(sample(orGate,1000))) // => 1
predictBasedOnNetWeights([0,1], train(sample(orGate,1000))) // => 1
predictBasedOnNetWeights([0,0], train(sample(orGate,1000))) // => 0

predictBasedOnNetWeights([1,1], train(sample(andGate,1000))) // => 1
predictBasedOnNetWeights([0,1], train(sample(andGate,1000))) // => 0
predictBasedOnNetWeights([1,0], train(sample(andGate,1000))) // => 0
predictBasedOnNetWeights([0,0], train(sample(andGate,1000))) // => 0

// xor gates will not do very well, its not linearly sapratable IE you can't build a linear function to split based on input 1 or 0 results
predictBasedOnNetWeights([1,1], train(sample(xorGate,100000))) // => random {0 or 1}
