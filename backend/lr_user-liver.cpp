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
    
    int numFeatures = 8; 
    LogisticRegression model(numFeatures);

    std::vector<double> userFeatures;

    std::cout << "Enter Total_Bilirubin: ";
    double totalBilirubin;
    std::cin >> totalBilirubin;
    userFeatures.push_back(totalBilirubin);

    std::cout << "Enter Direct_Bilirubin: ";
    double DirectBilirubin;
    std::cin >> DirectBilirubin;
    userFeatures.push_back(DirectBilirubin);

    std::cout << "Enter Alkaline_Phosphotase: ";
    double AlkalinePhosphotase;
    std::cin >> AlkalinePhosphotase;
    userFeatures.push_back(AlkalinePhosphotase);

    std::cout << "Enter Alamine_Aminotransferase: ";
    double AlamineAminotransferase;
    std::cin >> AlamineAminotransferase;
    userFeatures.push_back(AlamineAminotransferase);

    std::cout << "Enter Aspartate_Aminotransferase: ";
    double AspartateAminotransferase;
    std::cin >> AspartateAminotransferase;
    userFeatures.push_back(AspartateAminotransferase);

    std::cout << "Enter Total_Protiens: ";
    double TotalProtiens;
    std::cin >> TotalProtiens;
    userFeatures.push_back(TotalProtiens);

    std::cout << "Enter Albumin: ";
    double Albumin;
    std::cin >> Albumin;
    userFeatures.push_back(Albumin);

    std::cout << "Enter Albumin_and_Globulin_Ratio: ";
    double AlbuminandGlobulinRatio;
    std::cin >> AlbuminandGlobulinRatio;
    userFeatures.push_back(AlbuminandGlobulinRatio);


    double prediction = model.predict(userFeatures);

    std::cout << "Predicted Outcome: " << (prediction >= 0.5 ? 1 : 0) << std::endl;

    std::cout << "Enter Pid: ";
    int pid;
    std::cin >> pid;

    // Store the user input and prediction, including "Pid," in a CSV file
    std::ofstream outputFile("liver_predictions.csv", std::ios::app); 
    if (outputFile.is_open()) {
        outputFile << pid << ",";
        for (double feature : userFeatures) {
            outputFile << feature << ",";
        }
        outputFile << (prediction >= 0.5 ? 1 : 0) << std::endl;  
        outputFile.close();
    } else {
        std::cerr << "Unable to open the output file." << std::endl;
        return 1;
    }

    return 0;
}
