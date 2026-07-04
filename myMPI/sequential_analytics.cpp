#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <fstream>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: ./sequential_analytics <dataset_size>\n";
        return 1;
    }

    int n = stoi(argv[1]);
    vector<double> data1(n);
    vector<double> data2(n);

    cout << "Generating " << n << " data points for two columns...\n";

    // Data Generation Timing
    auto start_gen = chrono::high_resolution_clock::now();
    mt19937_64 rng(42);
    uniform_real_distribution<double> dist(0.0, 10000.0);

    for (int i = 0; i < n; ++i) {
        data1[i] = dist(rng);
        data2[i] = dist(rng);
    }
    auto end_gen = chrono::high_resolution_clock::now();
    double time_gen = chrono::duration<double, milli>(end_gen - start_gen).count();
    cout << "Data generation completed in " << time_gen << " ms.\n\n";

    // Setup CSV Logging
    ofstream csv_file("sequential_results.csv", ios::app);
    csv_file << "Dataset_Size,Task,Time_ms\n";

    // Task 1: Basic Statistics (Mean, Variance, StdDev, Min, Max)
    auto start_t1 = chrono::high_resolution_clock::now();
    
    double sum = 0, min_val = data1[0], max_val = data1[0];
    for(double x : data1) {
        sum += x;
        if(x < min_val) min_val = x;
        if(x > max_val) max_val = x;
    }
    double mean = sum / n;
    
    double variance_sum = 0;
    for(double x : data1) {
        variance_sum += (x - mean) * (x - mean);
    }
    double variance = variance_sum / n;
    double stddev = sqrt(variance);

    auto end_t1 = chrono::high_resolution_clock::now();
    double time_t1 = chrono::duration<double, milli>(end_t1 - start_t1).count();
    cout << "[Task 1] Basic Stats | Mean: " << mean << " | StdDev: " << stddev << " | Min: " << min_val << " | Max: " << max_val << "\n";
    cout << " -> Time: " << time_t1 << " ms\n";
    csv_file << n << ",Basic_Statistics," << time_t1 << "\n";

    // Task 2: Histogram Generation
    auto start_t2 = chrono::high_resolution_clock::now();
    
    int num_bins = 10;
    vector<int> histogram(num_bins, 0);
    double bin_width = (max_val - min_val) / num_bins;
    
    for(double x : data1) {
        int bin_index = (x - min_val) / bin_width;
        if (bin_index >= num_bins) bin_index = num_bins - 1; // Catch edge case for max_val
        histogram[bin_index]++;
    }

    auto end_t2 = chrono::high_resolution_clock::now();
    double time_t2 = chrono::duration<double, milli>(end_t2 - start_t2).count();
    cout << "[Task 2] Histogram built with " << num_bins << " bins.\n";
    cout << " -> Time: " << time_t2 << " ms\n";
    csv_file << n << ",Histogram," << time_t2 << "\n";

    // Task 3: Sorting (Sequential Baseline using std::sort)
    auto start_t3 = chrono::high_resolution_clock::now();
    
    vector<double> sorted_data = data1; // Copy data so original remains unsorted for moving average
    sort(sorted_data.begin(), sorted_data.end());

    auto end_t3 = chrono::high_resolution_clock::now();
    double time_t3 = chrono::duration<double, milli>(end_t3 - start_t3).count();
    cout << "[Task 3] Sorting completed.\n";
    cout << " -> Time: " << time_t3 << " ms\n";
    csv_file << n << ",Sorting," << time_t3 << "\n";

    // Task 4: Pearson Correlation
    auto start_t4 = chrono::high_resolution_clock::now();
    
    double sum2 = 0;
    for(double y : data2) sum2 += y;
    double mean2 = sum2 / n;

    double numerator = 0, sum_sq_x = 0, sum_sq_y = 0;
    for (int i = 0; i < n; ++i) {
        double diff_x = data1[i] - mean;
        double diff_y = data2[i] - mean2;
        numerator += diff_x * diff_y;
        sum_sq_x += diff_x * diff_x;
        sum_sq_y += diff_y * diff_y;
    }
    double correlation = numerator / sqrt(sum_sq_x * sum_sq_y);

    auto end_t4 = chrono::high_resolution_clock::now();
    double time_t4 = chrono::duration<double, milli>(end_t4 - start_t4).count();
    cout << "[Task 4] Pearson Correlation: " << correlation << "\n";
    cout << " -> Time: " << time_t4 << " ms\n";
    csv_file << n << ",Pearson_Correlation," << time_t4 << "\n";

    // Task 5: Moving Average
    auto start_t5 = chrono::high_resolution_clock::now();
    
    int window_size = 5;
    vector<double> moving_avg(n - window_size + 1);
    double current_sum = 0;
    
    for (int i = 0; i < window_size; ++i) {
        current_sum += data1[i];
    }
    moving_avg[0] = current_sum / window_size;
    
    for (int i = window_size; i < n; ++i) {
        current_sum += data1[i] - data1[i - window_size];
        moving_avg[i - window_size + 1] = current_sum / window_size;
    }

    auto end_t5 = chrono::high_resolution_clock::now();
    double time_t5 = chrono::duration<double, milli>(end_t5 - start_t5).count();
    cout << "[Task 5] Moving Average computed (Window: " << window_size << ").\n";
    cout << " -> Time: " << time_t5 << " ms\n";
    csv_file << n << ",Moving_Average," << time_t5 << "\n";


    // Task 6: Outlier Detection (Z-Score)
    auto start_t6 = chrono::high_resolution_clock::now();
    
    int outlier_count = 0;
    double z_threshold = 3.0; // Standard threshold for outliers
    
    for(double x : data1) {
        double z_score = abs(x - mean) / stddev;
        if (z_score > z_threshold) {
            outlier_count++;
        }
    }

    auto end_t6 = chrono::high_resolution_clock::now();
    double time_t6 = chrono::duration<double, milli>(end_t6 - start_t6).count();
    cout << "[Task 6] Outlier Detection: Found " << outlier_count << " outliers (Z > 3.0).\n";
    cout << " -> Time: " << time_t6 << " ms\n";
    csv_file << n << ",Outlier_Detection," << time_t6 << "\n";

    csv_file.close();
    cout << "\nAll results logged to sequential_results.csv successfully.\n";

    return 0;
}