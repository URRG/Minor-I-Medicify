#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>

const double learningRate = 0.01;
const int numEpochs = 1000;

// sigmoid function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


class LogisticRegression {
public:
    std::vector<double> weights;
    double bias;

    LogisticRegression(int numFeatures) {
        weights.resize(numFeatures, 0.0);
        bias = 0.0;
    }

    double predict(std::vector<double>& features) {
        double z = bias;
        for (int i = 0; i < features.size(); i++) {
            z += weights[i] * features[i];
        }
        return sigmoid(z);
    }

    void train(std::vector<std::vector<double>>& X, std::vector<int>& y) {
        int numSamples = X.size();
        int numFeatures = X[0].size();

        for (int epoch = 0; epoch < numEpochs; epoch++) {
            for (int i = 0; i < numSamples; i++) {
                double prediction = predict(X[i]);
                double error = prediction - y[i];

                bias -= learningRate * error;
                for (int j = 0; j < numFeatures; j++) {
                    weights[j] -= learningRate * error * X[i][j];
                }
            }
        }
    }
};

int main() {
    std::ifstream file("liver.csv");  
    if (!file.is_open()) {
        std::cerr << "Unable to open the dataset file." << std::endl;
        return 1;
    }

    
    std::vector<std::vector<double>> X; 
    std::vector<int> y; 

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        std::string token;
        while (std::getline(iss, token, ',')) {
            try {
                row.push_back(std::stod(token));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid data: " << token << std::endl;
            }
        }
        if (row.size() > 0) {
            y.push_back(static_cast<int>(row.back()));
            row.pop_back();
            X.push_back(row);
        }
    }

    LogisticRegression model(X[0].size());

    model.train(X, y);

    // Make predictions
    std::vector<int> predictions;
    for (int i = 0; i < X.size(); i++) {
        double prediction = model.predict(X[i]);
        predictions.push_back(prediction >= 0.5 ? 1 : 0);
    }

    // Calculate accuracy
    int correct = 0;
    for (int i = 0; i < X.size(); i++) {
        if (predictions[i] == y[i]) {
            correct++;
        }
    }

    double accuracy = static_cast<double>(correct) / X.size() * 100.0;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    return 0;
}
