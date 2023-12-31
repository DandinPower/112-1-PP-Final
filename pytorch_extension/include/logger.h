#pragma once
#include <chrono>
#include <iostream>
#include <map>
#include <string>

typedef enum {
    BUILTIN = 0,
    PARALLEL_STRUCTURE = 1,
    OPENMP = 2,
    OPENMP_MEM_EFFI = 3,
    STD_THREAD = 4
} TestType;

/**
 * Converts a TestType enum value to its corresponding string representation.
 *
 * @param test_type The TestType enum value to convert.
 * @return The string representation of the TestType enum value.
 */
std::string testTypeToString(TestType test_type) {
    switch (test_type) {
        case BUILTIN:
            return "BUILTIN";
        case PARALLEL_STRUCTURE:
            return "PARALLEL_STRUCTURE";
        case OPENMP:
            return "OPENMP";
        case OPENMP_MEM_EFFI:
            return "OPENMP_MEM_EFFI";
        case STD_THREAD:
            return "STD_THREAD";
        default:
            return "UNKNOWN";
    }
}

/**
 * @brief Struct representing the configuration for a test, which will received
 * from python side.
 */
struct TestConfig {
    TestType test_type; /**< The type of test. */
    int num_threads;    /**< The number of threads to use. */
    int A_row;          /**< The number of rows in matrix A. */
    int A_col;          /**< The number of columns in matrix A. */
    int B_row;          /**< The number of rows in matrix B. */
    int B_col;          /**< The number of columns in matrix B. */
    float A_density;    /**< The density of matrix A. */
    float B_density;    /**< The density of matrix B. */
};

/**
 * @class HighPrecisionLogger
 * @brief A class for logging high precision test durations.
 */
class HighPrecisionLogger {
   private:
    std::map<std::string, std::chrono::high_resolution_clock::time_point>
        startTimes; /**< Map to store start times of tests */
    std::map<std::string, std::chrono::high_resolution_clock::time_point>
        endTimes; /**< Map to store end times of tests */

   public:
    /**
     * @brief Reset the logger by clearing the start times and end times.
     */
    void reset() {
        startTimes.clear();
        endTimes.clear();
    }

    /**
     * @brief Start recording the duration of a test.
     * @param testName The name of the test.
     */
    void startTest(const std::string& testName) {
        startTimes[testName] = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief End recording the duration of a test.
     * @param testName The name of the test.
     */
    void endTest(const std::string& testName) {
        endTimes[testName] = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief Show the logs of the recorded test durations and the properties of
     * the test configuration.
     * @param config The test configuration.
     */
    void showLogs(const TestConfig& config) {
        printf("----------------------------------------\n");
        printf(
            "A=[ %d x %d, %.1f], B=[ %d x %d, %.1f], num_threads= [ %d ], "
            "type= %s\n",
            config.A_row, config.A_col, config.A_density, config.B_row,
            config.B_col, config.B_density, config.num_threads,
            testTypeToString(config.test_type).c_str());

        std::cout << "Test, Duration(ms), Percentage(%)" << std::endl;

        // Calculate total duration
        double totalDuration = 0.0;
        for (const auto& pair : startTimes) {
            const std::string& testName = pair.first;
            auto startTime = pair.second;
            auto endTime = endTimes[testName];

            double duration =
                std::chrono::duration_cast<std::chrono::microseconds>(endTime -
                                                                    startTime)
                    .count() /
                1000.0;
            totalDuration += duration;
        }

        // Print each test's duration and percentage
    for (const auto& pair : startTimes) {
        const std::string& testName = pair.first;
        auto startTime = pair.second;
        auto endTime = endTimes[testName];

        double duration =
            std::chrono::duration_cast<std::chrono::microseconds>(endTime -
                                                                  startTime)
                .count() /
            1000.0;
        double percentage = (duration / totalDuration) * 100;
        printf("%s,%.5f,%.2f\n", testName.c_str(), duration, percentage);
    }
}
};

// global logger, please carefully to use it in multi-thread environment.
HighPrecisionLogger logger;