// naieve perceptron
// trying to find a function such as or(a,b) = a * w0 + b * w1 + w3 , a,b = { 0 or 1 }

function train(trainingSet) {
	var firstRecord = trainingSet[0];

	// number of weights should be the number of features and every record should have the same number of features so the first one will do
	// one more weight for bias
	var weights = [];
	for(var i = 0; i < firstRecord.features.length + 1; i++) {
		weights.push(Math.random());
	}

	for(record of trainingSet) {

		// one extra feature for bias
		var prediction = predictBasedOnWeights(record.features, weights);

		for(featureIndex in record.features) {
			weights[featureIndex] += record.features[featureIndex] * (record.result - prediction) * 0.1;
		}

		// for the last bais weight
		weights[weights.length -1] += (record.result - prediction) * 0.1;
	}

	return weights;
}

function predictBasedOnWeights(recordFeatures, weights) {
	
	// add 1 for bias
	var inputFeatures = recordFeatures.concat(1);

	var numberOfTestFeatures = inputFeatures.length;
	var predictedResult = 0;

	for(featureIndex in inputFeatures) {
		predictedResult += inputFeatures[featureIndex] * weights[featureIndex];
	}
	
	return predictedResult > 0 ? 1 : 0;
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

predictBasedOnWeights([1,1], train(sample(orGate,1000))) // => 1
predictBasedOnWeights([1,0], train(sample(orGate,1000))) // => 1
predictBasedOnWeights([0,1], train(sample(orGate,1000))) // => 1
predictBasedOnWeights([0,0], train(sample(orGate,1000))) // => 0

predictBasedOnWeights([1,1], train(sample(andGate,1000))) // => 1
predictBasedOnWeights([0,1], train(sample(andGate,1000))) // => 0
predictBasedOnWeights([1,0], train(sample(andGate,1000))) // => 0
predictBasedOnWeights([0,0], train(sample(andGate,1000))) // => 0

// xor gates will not do very well, its not linearly sapratable IE you can't build a linear function to split based on input 1 or 0 results
predictBasedOnWeights([1,1], train(sample(xorGate,1000))) // => random {0 or 1}
