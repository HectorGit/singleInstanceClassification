// System includes
#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <algorithm>
#include <stdint.h>
#include <limits.h>
#include <stdexcept>

// OpenNN includes

#include "opennn/opennn.h"
#include <opennn/multilayer_perceptron.h>
#include <opennn/perceptron_layer.h>
#include "training_strategy.h"
//#include "tests/*.h"
#include "variables.h"
//#include "statistics.h"
using namespace OpenNN;
using namespace std;

int main(void)
{

	try
	{	
		bool debug = false;
		/*LOAD A PREVIOUSLY TRAINED NEURAL NETWORK !*/
		tinyxml2::XMLDocument myDoc;
		myDoc.LoadFile("data/proceeds/neural_network.xml");
		if (debug) { cout << "found xml file" << endl; }
		if (debug) { cout << "creating neural network from xml file" << endl; }
		NeuralNetwork neural_network(myDoc);
		if (debug) { cout << "created neural_network from xml file" << endl; }

		/*READ THE INPUT FOR AN INCOMING INSTANCE (UN LABELED) - MUST FORMAT CORRECTLY */
		Vector<double> vector_input;
		vector_input.load("data/input/single_instance.dat");
		if (debug) { cout << vector_input.to_string() << "\n" << endl; }

		/*READ THE INPUT FOR AN INCOMING MATRIX OF INSTANCES - MUST FORMAT CORRECTLY*/
		Matrix<double> matrix_input;
		matrix_input.load("data/input/some_instances.dat");
		if (debug) { cout << "matrix input: " << matrix_input.to_string() << "\n" << endl; }

		/*COMPARING THE SIZE OF ATTRIBUTES FOR AN INSTANCE , 
		AND THE NUMBER OF ATTRIBUTES THE NEURAL NETWORK WAS TRAINED ON 
		THESE SHOULD MATCH!!!*/
		if (debug) { cout << "vector_input size :" << vector_input.size() << endl; }
		if (debug) { cout << "neural_network inputs number:" << neural_network.get_inputs_number() << endl; }

		/*CALCULATE THE ACTIVATION OF THE LAST LEVEL OF THE NEURAL NETWORK
		SINGLE INSTANCE*/
		Vector<double> output_vector = neural_network.calculate_outputs(vector_input);
		/*GET THE INDEX OF THE MAXIMUM ACTIVATION == RESULTING CLASSIFICATION !!!*/
		size_t myClassificationIndex = output_vector.calculate_maximal_index();

		/*CALCULATE THE ACTIVATION OF THE LAST LEVEL OF THE NEURAL NETWORK 
		MULTIPLE INSTANCES AT ONCE*/
		Matrix<double> output_matrix = neural_network.calculate_output_data(matrix_input);
		cout << "output matrix number of rows" << output_matrix.get_rows_number() << endl;
		Vector<double> maxIndices;
		for (size_t i = 0; i < output_matrix.get_rows_number(); i++) {
			maxIndices.emplace_back(output_matrix.arrange_row(i).calculate_maximal_index());//getcalculate_maximal_index();
		}

		/*SHOW RESULTS - FOR SINGLE INPUT INSTANCE - WHAT IS THE RESULTING CLASSIFICATION?*/
		if (debug) {
			cout << "\n SHOW RESULTS - FOR MULTIPLE INPUT INSTANCES - WHAT ARE THE RESULTING OUTPUT FROM NN? \n" << endl;
			cout << "\n neural_network result for vector_input: " << output_vector.to_string() << "\n\n" << endl;
		}
		cout << "\n ------------Resulting classifications--------- \n" << endl;
		cout << "\n For vector_input: " << myClassificationIndex << "\n" << endl;
		
		/*SHOW RESULTS - FOR MULTIPLE INPUT INSTANCES - WHAT ARE THE RESULTING CLASSIFICATIONS?*/
		//if (debug) {
			cout << "\n SHOW RESULTS - FOR MULTIPLE INPUT INSTANCES - WHAT ARE THE RESULTING OUTPUT FROM NN? \n" << endl;
			cout << "\n neural_network result for matrix_input:\n " << output_matrix.to_string() << "\n\n" << endl;
		//}
		cout << "\n For matrix_input: " << maxIndices.to_string() << "\n" << endl;

		cin.get();
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return(1);
	}

}