#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <mpi.h>

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) cout << "Usage: mpiexec -n 4 mpi_analytics.exe <dataset_size>\n";
        MPI_Finalize();
        return 1;
    }

    int n = stoi(argv[1]);
    int local_n = n / size;

    vector<double> data1, data2;
    vector<double> local_data1(local_n), local_data2(local_n);

    // Data Generation & Setup (Master Node)
    if (rank == 0) {
        data1.resize(n);
        data2.resize(n);
        cout << "Generating " << n << " data points on Master Node...\n";
        
        mt19937_64 rng(42);
        uniform_real_distribution<double> dist(0.0, 10000.0);
        for (int i = 0; i < n; ++i) {
            data1[i] = dist(rng);
            data2[i] = dist(rng);
        }
    }

    ofstream csv_file;
    if (rank == 0) {
        csv_file.open("mpi_results.csv", ios::app);
        csv_file << "Dataset_Size,Task,Time_ms\n";
    }

    // Data Distribution
    MPI_Barrier(MPI_COMM_WORLD);
    double start_total = MPI_Wtime();

    MPI_Scatter(data1.data(), local_n, MPI_DOUBLE, local_data1.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(data2.data(), local_n, MPI_DOUBLE, local_data2.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Task 1: Basic Statistics
    MPI_Barrier(MPI_COMM_WORLD);
    double task1_start = MPI_Wtime();
    
    double local_sum = 0, local_min = local_data1[0], local_max = local_data1[0];
    for(double x : local_data1) {
        local_sum += x;
        if(x < local_min) local_min = x;
        if(x > local_max) local_max = x;
    }

    double global_sum = 0, global_min = 0, global_max = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double global_mean = 0;
    if (rank == 0) global_mean = global_sum / n;
    MPI_Bcast(&global_mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double local_var_sum = 0;
    for(double x : local_data1) local_var_sum += (x - global_mean) * (x - global_mean);
    
    double global_var_sum = 0;
    MPI_Reduce(&local_var_sum, &global_var_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double global_stddev = 0;
    if (rank == 0) {
        global_stddev = sqrt(global_var_sum / n);
        double task1_time = (MPI_Wtime() - task1_start) * 1000.0;
        cout << "[Task 1] Mean: " << global_mean << " | StdDev: " << global_stddev << " | Time: " << task1_time << " ms\n";
        csv_file << n << ",Basic_Statistics," << task1_time << "\n";
    }
    MPI_Bcast(&global_stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Task 2: Histogram Generation
    MPI_Barrier(MPI_COMM_WORLD);
    double task2_start = MPI_Wtime();
    
    int num_bins = 10;
    vector<int> local_hist(num_bins, 0);
    vector<int> global_hist(num_bins, 0);
    double bin_width = 10000.0 / num_bins; // based on distribution max
    
    for(double x : local_data1) {
        int bin = x / bin_width;
        if (bin >= num_bins) bin = num_bins - 1;
        local_hist[bin]++;
    }
    
    MPI_Reduce(local_hist.data(), global_hist.data(), num_bins, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double task2_time = (MPI_Wtime() - task2_start) * 1000.0;
        cout << "[Task 2] Histogram built. | Time: " << task2_time << " ms\n";
        csv_file << n << ",Histogram," << task2_time << "\n";
    }

    // Task 3: Parallel Sorting (Block Gather Merge)
    MPI_Barrier(MPI_COMM_WORLD);
    double task3_start = MPI_Wtime();
    
    vector<double> local_sorted = local_data1;
    sort(local_sorted.begin(), local_sorted.end());
    
    vector<double> global_sorted;
    if (rank == 0) global_sorted.resize(n);
    
    MPI_Gather(local_sorted.data(), local_n, MPI_DOUBLE, global_sorted.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        sort(global_sorted.begin(), global_sorted.end()); // Master merges blocks
        double task3_time = (MPI_Wtime() - task3_start) * 1000.0;
        cout << "[Task 3] Sorting completed. | Time: " << task3_time << " ms\n";
        csv_file << n << ",Sorting," << task3_time << "\n";
    }

    // Task 4: Pearson Correlation
    MPI_Barrier(MPI_COMM_WORLD);
    double task4_start = MPI_Wtime();
    
    double local_sum2 = 0;
    for(double y : local_data2) local_sum2 += y;
    
    double global_sum2 = 0;
    MPI_Reduce(&local_sum2, &global_sum2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double global_mean2 = 0;
    if (rank == 0) global_mean2 = global_sum2 / n;
    MPI_Bcast(&global_mean2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double local_num = 0, local_sq_x = 0, local_sq_y = 0;
    for (int i = 0; i < local_n; ++i) {
        double dx = local_data1[i] - global_mean;
        double dy = local_data2[i] - global_mean2;
        local_num += dx * dy;
        local_sq_x += dx * dx;
        local_sq_y += dy * dy;
    }

    double global_num = 0, global_sq_x = 0, global_sq_y = 0;
    MPI_Reduce(&local_num, &global_num, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sq_x, &global_sq_x, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sq_y, &global_sq_y, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double correlation = global_num / sqrt(global_sq_x * global_sq_y);
        double task4_time = (MPI_Wtime() - task4_start) * 1000.0;
        cout << "[Task 4] Pearson Correlation: " << correlation << " | Time: " << task4_time << " ms\n";
        csv_file << n << ",Pearson_Correlation," << task4_time << "\n";
    }

    // Task 5: Moving Average (Local Chunks)
    MPI_Barrier(MPI_COMM_WORLD);
    double task5_start = MPI_Wtime();
    
    int window = 5;
    vector<double> local_ma(local_n - window + 1);
    double cur_sum = 0;
    for (int i = 0; i < window; ++i) cur_sum += local_data1[i];
    local_ma[0] = cur_sum / window;
    
    for (int i = window; i < local_n; ++i) {
        cur_sum += local_data1[i] - local_data1[i - window];
        local_ma[i - window + 1] = cur_sum / window;
    }
    
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before timing check
    if (rank == 0) {
        double task5_time = (MPI_Wtime() - task5_start) * 1000.0;
        cout << "[Task 5] Moving Average computed. | Time: " << task5_time << " ms\n";
        csv_file << n << ",Moving_Average," << task5_time << "\n";
    }

    // Task 6: Outlier Detection
    MPI_Barrier(MPI_COMM_WORLD);
    double task6_start = MPI_Wtime();
    
    int local_outliers = 0;
    for(double x : local_data1) {
        if (abs(x - global_mean) / global_stddev > 3.0) local_outliers++;
    }

    int global_outliers = 0;
    MPI_Reduce(&local_outliers, &global_outliers, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double task6_time = (MPI_Wtime() - task6_start) * 1000.0;
        cout << "[Task 6] Outliers Found: " << global_outliers << " | Time: " << task6_time << " ms\n";
        csv_file << n << ",Outlier_Detection," << task6_time << "\n";
    }

    // Wrap up
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        double total_time = (MPI_Wtime() - start_total) * 1000.0;
        cout << "\nTotal MPI Parallel Time: " << total_time << " ms\n";
        csv_file << n << ",Total_Execution," << total_time << "\n\n";
        csv_file.close();
        cout << "Results logged to mpi_results.csv successfully.\n";
    }

    MPI_Finalize();
    return 0;
}