#ifndef GPU_PARTITIONER_H
#define GPU_PARTITIONER_H

#include <vector>
#include <map>
#include <set>
#include <numeric> // for std::accumulate
#include <cmath>   // for std::pow
#include <algorithm> // for std::lexicographical_compare
#include <iostream> // for std::cout
#include <sstream>  // for std::ostringstream
#include <string>   // for std::string

// Represents a contiguous region of GPU memory
int A100_GPC_SLICE_NUM = 7;
struct MemoryPartition {
    int size_in_slices;
    bool allocated;

    // comparison needed for std::set or std::map
    bool operator<(const MemoryPartition& other) const {
        if (size_in_slices != other.size_in_slices) {
            return size_in_slices < other.size_in_slices;
        }
        // allocated = true (1) should come after allocated = false (0)
        return allocated < other.allocated;
    }
};

// Helper struct to store block start index and size
struct BlockInfo {
    int start_slice;
    int size_in_slices;

    bool operator==(const BlockInfo& other) const {
        return start_slice == other.start_slice && size_in_slices == other.size_in_slices;
    }

    // Add operator!= based on operator==
    bool operator!=(const BlockInfo& other) const {
        return !(*this == other);
    }
};

// Helper function to convert partition representations to BlockInfo vectors
std::vector<BlockInfo> get_block_info(const std::vector<MemoryPartition>& state) {
    std::vector<BlockInfo> blocks;
    int current_slice = 0;
    for (const auto& partition : state) {
        if (partition.allocated) {
            blocks.push_back({current_slice, partition.size_in_slices});
        }
        current_slice += partition.size_in_slices;
    }
    return blocks;
}

std::vector<BlockInfo> get_block_info(const std::vector<int>& config) {
    std::vector<BlockInfo> blocks;
    int current_slice = 0;
    for (int size : config) {
        blocks.push_back({current_slice, size});
        current_slice += size;
    }
    return blocks;
}

// Feasible full partition configurations for a GPU with 7 GPC slices (based on the provided image)
// Each inner vector represents a configuration, and the integers are the sizes of contiguous partitions.
const std::vector<std::vector<int>> gpc_slice_configurations = {
    {7},             // Config 1
    {4, 3},          // Config 2
    {4, 2, 1},       // Config 3
    {4, 1, 1, 1},    // Config 4
    {3, 3},          // Config 5 (Assumes total 6 slices usable in this config)
    {3, 2, 1},       // Config 6 (Assumes total 6 slices usable)
    {3, 1, 1, 1},    // Config 7 (Assumes total 6 slices usable)
    {2, 2, 3},       // Config 8
    {2, 1, 1, 3},    // Config 9
    {1, 1, 2, 3},    // Config 10
    {1, 1, 1, 1, 3}, // Config 11
    {2, 2, 2, 1},    // Config 12
    {2, 1, 1, 2, 1}, // Config 13
    {1, 1, 2, 2, 1}, // Config 14
    {2, 1, 1, 1, 1, 1}, // Config 15
    {1, 1, 2, 1, 1, 1}, // Config 16
    {1, 1, 1, 1, 2, 1}, // Config 17
    {1, 1, 1, 1, 1, 2}, // Config 18
    {1, 1, 1, 1, 1, 1, 1}  // Config 19
};

/**
 * @brief Counts how many known full configurations are reachable from a given partial state.
 *
 * A full configuration is reachable if all allocated blocks in the partial state
 * exist with the exact same start slice and size in the full configuration.
 *
 * @param partial_state The current state of GPU partitions (allocated and unallocated).
 * @param full_configurations A list of known valid full configurations.
 * @return The number of full configurations reachable from the partial state.
 */
int count_reachable_configurations(
    const std::vector<MemoryPartition>& partial_state,
    const std::vector<std::vector<int>>& full_configurations = gpc_slice_configurations)
{
    int reachable_count = 0;
    std::vector<BlockInfo> allocated_blocks = get_block_info(partial_state);

    if (allocated_blocks.empty()) {
        // If there are no allocated blocks yet, any full configuration is potentially reachable
        // depending on whether the *total* size matches. Here, we assume all listed configs
        // are for the target total size (e.g., 7 slices), so all are reachable.
        // A more robust check might compare total slice counts.
        return full_configurations.size();
    }

    for (const auto& full_config : full_configurations) {
        std::vector<BlockInfo> full_blocks = get_block_info(full_config);

        // Check if allocated_blocks is a subsequence of full_blocks
        int alloc_idx = 0;
        int full_idx = 0;
        while (alloc_idx < allocated_blocks.size() && full_idx < full_blocks.size()) {
            // Find the next block in full_blocks that matches the current allocated_block
            while (full_idx < full_blocks.size() && full_blocks[full_idx] != allocated_blocks[alloc_idx]) {
                full_idx++;
            }

            // If a match was found, advance both pointers
            if (full_idx < full_blocks.size()) {
                alloc_idx++;
                full_idx++; // Move to the next block in full_blocks for the next search
            }
            // If no match found for allocated_blocks[alloc_idx], this full_config is not reachable
        }

        // If all allocated blocks were found in order, the config is reachable
        if (alloc_idx == allocated_blocks.size()) {
            reachable_count++;
        }
    }

    return reachable_count;
}

// Helper to convert a state vector to a string for printing
std::string to_string(const std::vector<MemoryPartition>& state) {
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (const auto& p : state) {
        if (!first) {
            oss << ", ";
        }
        oss << p.size_in_slices << ":" << (p.allocated ? 'A' : 'U');
        first = false;
    }
    oss << "]";
    return oss.str();
}

// Type alias for the cache map
using ReachabilityCache = std::map<std::vector<MemoryPartition>, int>;

/**
 * @brief Merges adjacent unallocated partitions in a state vector.
 *
 * @param state The state vector to normalize.
 * @return std::vector<MemoryPartition> The normalized state vector.
 */
std::vector<MemoryPartition> normalize_state(const std::vector<MemoryPartition>& state) {
    if (state.empty()) {
        return {};
    }
    std::vector<MemoryPartition> normalized;
    normalized.push_back(state[0]);
    for (size_t i = 1; i < state.size(); ++i) {
        if (!normalized.back().allocated && !state[i].allocated) {
            // Merge current unallocated block into the previous one
            normalized.back().size_in_slices += state[i].size_in_slices;
        } else {
            normalized.push_back(state[i]);
        }
    }

    // Trim trailing unallocated block if it exists and is not the only block
    if (normalized.size() > 1 && !normalized.back().allocated) {
        normalized.pop_back();
    }

    return normalized;
}


void generate_integer_partitions(int remaining_slices, std::vector<int>& current_partition, std::set<std::vector<int>>& all_partitions) {
    if (remaining_slices == 0) {
        all_partitions.insert(current_partition);
        return;
    }
    for(int next_partition_size = 1; next_partition_size <= remaining_slices; ++next_partition_size) {
        current_partition.push_back(next_partition_size);
        generate_integer_partitions(remaining_slices - next_partition_size, current_partition, all_partitions);
        current_partition.pop_back();
    }
}
/**
 * @brief Precomputes the number of reachable full configurations for all possible
 *        partial states derivable from the known full configurations.
 *
 * @param full_configurations The list of known valid full configurations.
 * @return ReachabilityCache A map where keys are unique partial states and
 *         values are the count of full configurations reachable from that state.
 */
ReachabilityCache precompute_reachable_counts(
    const std::vector<std::vector<int>>& full_configurations = gpc_slice_configurations)
{
    std::set<std::vector<int>> all_partitions;
    std::vector<int> current_partition;
    generate_integer_partitions(A100_GPC_SLICE_NUM, current_partition, all_partitions);

    std::set<std::vector<MemoryPartition>> unique_partial_states;
    ReachabilityCache cache;

    for(const auto& partition : all_partitions) {
        int num_blocks = partition.size();
        if (num_blocks == 0) continue; // Should not happen if A100_GPC_SLICE_NUM > 0

        long long num_alloc_combinations = 1LL << num_blocks;

        // Enumerate all possible true/false combinations for the partition blocks
        for (long long i = 0; i < num_alloc_combinations; ++i) {
            std::vector<MemoryPartition> current_state_combination;
            long long current_bitmask = i;
            for (int block_size : partition) {
                 bool is_allocated = (current_bitmask & 1);
                 current_state_combination.push_back({block_size, is_allocated});
                 current_bitmask >>= 1;
            }

            // Normalize the state
            std::vector<MemoryPartition> normalized_state = normalize_state(current_state_combination);

            // Insert the unique normalized state into the set
             if (!normalized_state.empty()) {
                 unique_partial_states.insert(normalized_state);
             }
        }
    }

    // Ensure the fully unallocated state is present 
    // (Could be missed if A100_GPC_SLICE_NUM=0 or if {k} partition wasn't generated)
    unique_partial_states.insert(normalize_state({{A100_GPC_SLICE_NUM, false}}));


    // Now, compute the reachability for each unique state found
    std::cout << "Generated " << all_partitions.size() << " base integer partitionings." << std::endl;
    std::cout << "Generated Unique Partial States (" << unique_partial_states.size() << ") using integer partitioning enumeration:\\n";
    for (const auto& partial_state : unique_partial_states) {
        // We still use the original full_configurations to check reachability
        int reachable_count = count_reachable_configurations(partial_state, full_configurations);
        if (reachable_count > 0) {
            cache[partial_state] = reachable_count;
            std::cout << "  " << to_string(partial_state) << " -> Reachable: " << reachable_count << std::endl;
        }
    }
    std::cout << "----------------------------------\\n";

    return cache;
}

// Hardcoded cache generated from precomputation (Integer Partitioning Enumeration)
static const ReachabilityCache precomputed_reachability_cache = {
  { {{1, false}, {1, true}}, 7 },
  { {{1, false}, {1, true}, {1, false}, {1, true}}, 4 },
  { {{1, false}, {1, true}, {1, false}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, false}, {1, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, false}, {1, true}, {1, false}, {2, true}}, 1 },
  { {{1, false}, {1, true}, {1, false}, {1, true}, {1, true}}, 2 },
  { {{1, false}, {1, true}, {1, false}, {1, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, false}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, false}, {1, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, false}, {1, true}, {1, true}, {2, true}}, 1 },
  { {{1, false}, {1, true}, {1, false}, {1, true}, {2, false}, {1, true}}, 2 },
  { {{1, false}, {1, true}, {1, false}, {1, true}, {2, true}}, 1 },
  { {{1, false}, {1, true}, {1, false}, {1, true}, {2, true}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, false}, {1, true}, {3, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}}, 4 },
  { {{1, false}, {1, true}, {1, true}, {1, false}, {1, true}}, 2 },
  { {{1, false}, {1, true}, {1, true}, {1, false}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {1, false}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {1, false}, {1, true}, {2, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {1, false}, {2, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {1, false}, {2, true}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {1, false}, {3, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {1, true}}, 4 },
  { {{1, false}, {1, true}, {1, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {1, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {1, true}, {1, false}, {2, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {1, true}, {1, true}}, 2 },
  { {{1, false}, {1, true}, {1, true}, {1, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {1, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {1, true}, {1, true}, {2, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {1, true}, {2, false}, {1, true}}, 2 },
  { {{1, false}, {1, true}, {1, true}, {1, true}, {2, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {1, true}, {2, true}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {1, true}, {3, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {2, false}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {2, false}, {1, true}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {2, false}, {2, true}}, 1 },
  { {{1, false}, {1, true}, {1, true}, {3, false}, {1, true}}, 2 },
  { {{1, false}, {1, true}, {2, false}, {1, true}}, 3 },
  { {{1, false}, {1, true}, {2, false}, {1, true}, {1, false}, {1, true}}, 2 },
  { {{1, false}, {1, true}, {2, false}, {1, true}, {1, true}}, 2 },
  { {{1, false}, {1, true}, {2, false}, {1, true}, {1, true}, {1, true}}, 2 },
  { {{1, false}, {1, true}, {2, false}, {1, true}, {2, true}}, 1 },
  { {{1, false}, {1, true}, {2, false}, {2, true}}, 2 },
  { {{1, false}, {1, true}, {2, false}, {2, true}, {1, true}}, 2 },
  { {{1, false}, {1, true}, {2, false}, {3, true}}, 2 },
  { {{1, false}, {1, true}, {2, true}}, 3 },
  { {{1, false}, {1, true}, {2, true}, {1, false}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {2, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {2, true}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {2, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {2, true}, {1, true}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {2, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {2, true}, {2, false}, {1, true}}, 2 },
  { {{1, false}, {1, true}, {2, true}, {2, true}}, 1 },
  { {{1, false}, {1, true}, {2, true}, {2, true}, {1, true}}, 1 },
  { {{1, false}, {1, true}, {2, true}, {3, true}}, 1 },
  { {{1, false}, {1, true}, {3, false}, {1, true}}, 2 },
  { {{1, false}, {1, true}, {3, false}, {1, true}, {1, true}}, 2 },
  { {{1, false}, {1, true}, {3, false}, {2, true}}, 1 },
  { {{1, false}, {1, true}, {4, false}, {1, true}}, 4 },
  { {{1, true}}, 7 },
  { {{1, true}, {1, false}, {1, true}}, 4 },
  { {{1, true}, {1, false}, {1, true}, {1, false}, {1, true}}, 2 },
  { {{1, true}, {1, false}, {1, true}, {1, false}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {1, false}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {1, false}, {1, true}, {2, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {1, false}, {2, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {1, false}, {2, true}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {1, false}, {3, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {1, true}}, 4 },
  { {{1, true}, {1, false}, {1, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {1, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {1, true}, {1, false}, {2, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {1, true}, {1, true}}, 2 },
  { {{1, true}, {1, false}, {1, true}, {1, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {1, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {1, true}, {1, true}, {2, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {1, true}, {2, false}, {1, true}}, 2 },
  { {{1, true}, {1, false}, {1, true}, {1, true}, {2, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {1, true}, {2, true}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {1, true}, {3, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {2, false}, {1, true}}, 1 }, 
  { {{1, true}, {1, false}, {1, true}, {2, false}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {2, false}, {2, true}}, 1 },
  { {{1, true}, {1, false}, {1, true}, {3, false}, {1, true}}, 2 },
  { {{1, true}, {1, false}, {2, true}}, 3 },
  { {{1, true}, {1, false}, {2, true}, {1, false}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {2, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {2, true}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {2, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {2, true}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {2, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {2, true}, {2, false}, {1, true}}, 2 },
  { {{1, true}, {1, false}, {2, true}, {2, true}}, 1 },
  { {{1, true}, {1, false}, {2, true}, {2, true}, {1, true}}, 1 },
  { {{1, true}, {1, false}, {2, true}, {3, true}}, 1 },
  { {{1, true}, {1, true}}, 7 },
  { {{1, true}, {1, true}, {1, false}, {1, true}}, 4 },
  { {{1, true}, {1, true}, {1, false}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, false}, {1, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, false}, {1, true}, {1, false}, {2, true}}, 1 },
  { {{1, true}, {1, true}, {1, false}, {1, true}, {1, true}}, 2 },
  { {{1, true}, {1, true}, {1, false}, {1, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, false}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, false}, {1, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, false}, {1, true}, {1, true}, {2, true}}, 1 },
  { {{1, true}, {1, true}, {1, false}, {1, true}, {2, false}, {1, true}}, 2 },
  { {{1, true}, {1, true}, {1, false}, {1, true}, {2, true}}, 1 },
  { {{1, true}, {1, true}, {1, false}, {1, true}, {2, true}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, false}, {1, true}, {3, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}}, 4 },
  { {{1, true}, {1, true}, {1, true}, {1, false}, {1, true}}, 2 },
  { {{1, true}, {1, true}, {1, true}, {1, false}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {1, false}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {1, false}, {1, true}, {2, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {1, false}, {2, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {1, false}, {2, true}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {1, false}, {3, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {1, true}}, 4 },
  { {{1, true}, {1, true}, {1, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {1, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {1, true}, {1, false}, {2, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {1, true}, {1, true}}, 2 },
  { {{1, true}, {1, true}, {1, true}, {1, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {1, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {1, true}, {1, true}, {2, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {1, true}, {2, false}, {1, true}}, 2 },
  { {{1, true}, {1, true}, {1, true}, {1, true}, {2, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {1, true}, {2, true}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {1, true}, {3, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {2, false}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {2, false}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {2, false}, {2, true}}, 1 },
  { {{1, true}, {1, true}, {1, true}, {3, false}, {1, true}}, 2 },
  { {{1, true}, {1, true}, {2, false}, {1, true}}, 3 },
  { {{1, true}, {1, true}, {2, false}, {1, true}, {1, false}, {1, true}}, 2 },
  { {{1, true}, {1, true}, {2, false}, {1, true}, {1, true}}, 2 },
  { {{1, true}, {1, true}, {2, false}, {1, true}, {1, true}, {1, true}}, 2 },
  { {{1, true}, {1, true}, {2, false}, {1, true}, {2, true}}, 1 },
  { {{1, true}, {1, true}, {2, false}, {2, true}}, 2 },
  { {{1, true}, {1, true}, {2, false}, {2, true}, {1, true}}, 2 },
  { {{1, true}, {1, true}, {2, false}, {3, true}}, 2 },
  { {{1, true}, {1, true}, {2, true}}, 3 },
  { {{1, true}, {1, true}, {2, true}, {1, false}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {2, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {2, true}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {2, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {2, true}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {2, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {2, true}, {2, false}, {1, true}}, 2 },
  { {{1, true}, {1, true}, {2, true}, {2, true}}, 1 },
  { {{1, true}, {1, true}, {2, true}, {2, true}, {1, true}}, 1 },
  { {{1, true}, {1, true}, {2, true}, {3, true}}, 1 },
  { {{1, true}, {1, true}, {3, false}, {1, true}}, 2 },
  { {{1, true}, {1, true}, {3, false}, {1, true}, {1, true}}, 2 },
  { {{1, true}, {1, true}, {3, false}, {2, true}}, 1 },
  { {{1, true}, {1, true}, {4, false}, {1, true}}, 4 },
  { {{1, true}, {2, false}, {1, true}}, 4 },
  { {{1, true}, {2, false}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, true}, {2, false}, {1, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {2, false}, {1, true}, {1, false}, {2, true}}, 1 },
  { {{1, true}, {2, false}, {1, true}, {1, true}}, 2 },
  { {{1, true}, {2, false}, {1, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{1, true}, {2, false}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {2, false}, {1, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{1, true}, {2, false}, {1, true}, {1, true}, {2, true}}, 1 },
  { {{1, true}, {2, false}, {1, true}, {2, false}, {1, true}}, 2 },
  { {{1, true}, {2, false}, {1, true}, {2, true}}, 1 },
  { {{1, true}, {2, false}, {1, true}, {2, true}, {1, true}}, 1 },
  { {{1, true}, {2, false}, {1, true}, {3, true}}, 1 },
  { {{1, true}, {3, false}, {1, true}}, 3 },
  { {{1, true}, {3, false}, {1, true}, {1, false}, {1, true}}, 2 },
  { {{1, true}, {3, false}, {1, true}, {1, true}}, 2 },
  { {{1, true}, {3, false}, {1, true}, {1, true}, {1, true}}, 2 },
  { {{1, true}, {3, false}, {1, true}, {2, true}}, 1 },
  { {{1, true}, {3, false}, {2, true}}, 2 },
  { {{1, true}, {3, false}, {2, true}, {1, true}}, 2 },
  { {{1, true}, {3, false}, {3, true}}, 2 },
  { {{1, true}, {4, false}, {1, true}}, 2 },
  { {{1, true}, {4, false}, {1, true}, {1, true}}, 2 },
  { {{1, true}, {4, false}, {2, true}}, 1 },
  { {{1, true}, {5, false}, {1, true}}, 4 },
  { {{2, false}, {1, true}}, 7 },
  { {{2, false}, {1, true}, {1, false}, {1, true}}, 3 },
  { {{2, false}, {1, true}, {1, false}, {1, true}, {1, false}, {1, true}}, 2 },
  { {{2, false}, {1, true}, {1, false}, {1, true}, {1, true}}, 2 },
  { {{2, false}, {1, true}, {1, false}, {1, true}, {1, true}, {1, true}}, 2 },
  { {{2, false}, {1, true}, {1, false}, {1, true}, {2, true}}, 1 },
  { {{2, false}, {1, true}, {1, false}, {2, true}}, 2 },
  { {{2, false}, {1, true}, {1, false}, {2, true}, {1, true}}, 2 },
  { {{2, false}, {1, true}, {1, false}, {3, true}}, 2 },
  { {{2, false}, {1, true}, {1, true}}, 7 },
  { {{2, false}, {1, true}, {1, true}, {1, false}, {1, true}}, 2 },
  { {{2, false}, {1, true}, {1, true}, {1, false}, {1, true}, {1, true}}, 2 },
  { {{2, false}, {1, true}, {1, true}, {1, false}, {2, true}}, 1 },
  { {{2, false}, {1, true}, {1, true}, {1, true}}, 3 },
  { {{2, false}, {1, true}, {1, true}, {1, true}, {1, false}, {1, true}}, 2 },
  { {{2, false}, {1, true}, {1, true}, {1, true}, {1, true}}, 2 },
  { {{2, false}, {1, true}, {1, true}, {1, true}, {1, true}, {1, true}}, 2 },
  { {{2, false}, {1, true}, {1, true}, {1, true}, {2, true}}, 1 },
  { {{2, false}, {1, true}, {1, true}, {2, false}, {1, true}}, 4 },
  { {{2, false}, {1, true}, {1, true}, {2, true}}, 2 },
  { {{2, false}, {1, true}, {1, true}, {2, true}, {1, true}}, 2 },
  { {{2, false}, {1, true}, {1, true}, {3, true}}, 2 },
  { {{2, false}, {1, true}, {2, false}, {1, true}}, 2 },
  { {{2, false}, {1, true}, {2, false}, {1, true}, {1, true}}, 2 },
  { {{2, false}, {1, true}, {2, false}, {2, true}}, 1 },
  { {{2, false}, {1, true}, {3, false}, {1, true}}, 4 },
  { {{2, false}, {2, true}}, 5 },
  { {{2, false}, {2, true}, {1, false}, {1, true}}, 1 },
  { {{2, false}, {2, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{2, false}, {2, true}, {1, true}}, 1 },
  { {{2, false}, {2, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{2, false}, {2, true}, {1, true}, {1, true}}, 1 },
  { {{2, false}, {2, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{2, false}, {2, true}, {2, false}, {1, true}}, 3 },
  { {{2, false}, {2, true}, {2, true}}, 2 },
  { {{2, false}, {2, true}, {2, true}, {1, true}}, 2 },
  { {{2, false}, {2, true}, {3, true}}, 2 },
  { {{2, true}}, 5 },
  { {{2, true}, {1, false}, {1, true}}, 3 },
  { {{2, true}, {1, false}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{2, true}, {1, false}, {1, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{2, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{2, true}, {1, false}, {1, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{2, true}, {1, false}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{2, true}, {1, false}, {1, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{2, true}, {1, false}, {1, true}, {2, false}, {1, true}}, 2 },
  { {{2, true}, {1, false}, {1, true}, {2, true}}, 1 },
  { {{2, true}, {1, false}, {1, true}, {2, true}, {1, true}}, 1 },
  { {{2, true}, {1, false}, {1, true}, {3, true}}, 1 },
  { {{2, true}, {1, true}}, 3 },
  { {{2, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{2, true}, {1, true}, {1, false}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{2, true}, {1, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{2, true}, {1, true}, {1, false}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{2, true}, {1, true}, {1, false}, {2, true}}, 1 },
  { {{2, true}, {1, true}, {1, false}, {2, true}, {1, true}}, 1 },
  { {{2, true}, {1, true}, {1, false}, {3, true}}, 1 },
  { {{2, true}, {1, true}, {1, true}}, 3 },
  { {{2, true}, {1, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{2, true}, {1, true}, {1, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{2, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{2, true}, {1, true}, {1, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{2, true}, {1, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{2, true}, {1, true}, {1, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{2, true}, {1, true}, {1, true}, {2, false}, {1, true}}, 2 },
  { {{2, true}, {1, true}, {1, true}, {2, true}}, 1 },
  { {{2, true}, {1, true}, {1, true}, {2, true}, {1, true}}, 1 },
  { {{2, true}, {1, true}, {1, true}, {3, true}}, 1 },
  { {{2, true}, {1, true}, {2, false}, {1, true}}, 1 },
  { {{2, true}, {1, true}, {2, false}, {1, true}, {1, true}}, 1 },
  { {{2, true}, {1, true}, {3, false}, {1, true}}, 2 },
  { {{2, true}, {2, false}, {1, true}}, 1 },
  { {{2, true}, {2, false}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{2, true}, {2, false}, {1, true}, {1, true}}, 1 },
  { {{2, true}, {2, false}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{2, true}, {2, false}, {2, true}}, 2 },
  { {{2, true}, {2, false}, {2, true}, {1, true}}, 2 },
  { {{2, true}, {2, false}, {3, true}}, 2 },
  { {{2, true}, {2, true}}, 2 },
  { {{2, true}, {2, true}, {2, false}, {1, true}}, 1 },
  { {{2, true}, {2, true}, {2, true}}, 1 },
  { {{2, true}, {2, true}, {2, true}, {1, true}}, 1 },
  { {{2, true}, {2, true}, {3, true}}, 1 },
  { {{2, true}, {3, false}, {1, true}}, 1 },
  { {{2, true}, {3, false}, {1, true}, {1, true}}, 1 },
  { {{2, true}, {4, false}, {1, true}}, 3 },
  { {{3, false}, {1, true}}, 8 },
  { {{3, false}, {1, true}, {1, false}, {1, true}}, 3 },
  { {{3, false}, {1, true}, {1, false}, {1, true}, {1, true}}, 2 },
  { {{3, false}, {1, true}, {1, false}, {2, true}}, 1 },
  { {{3, false}, {1, true}, {1, true}}, 4 },
  { {{3, false}, {1, true}, {1, true}, {1, false}, {1, true}}, 2 },
  { {{3, false}, {1, true}, {1, true}, {1, true}}, 3 },
  { {{3, false}, {1, true}, {1, true}, {1, true}, {1, true}}, 2 },
  { {{3, false}, {1, true}, {1, true}, {2, true}}, 1 },
  { {{3, false}, {1, true}, {2, false}, {1, true}}, 4 },
  { {{3, false}, {1, true}, {2, true}}, 2 },
  { {{3, false}, {1, true}, {2, true}, {1, true}}, 2 },
  { {{3, false}, {1, true}, {3, true}}, 2 },
  { {{3, false}, {2, true}}, 1 },
  { {{3, false}, {2, true}, {1, true}}, 1 },
  { {{3, false}, {3, true}}, 1 },
  { {{3, true}}, 3 },
  { {{3, true}, {1, false}, {1, true}}, 1 },
  { {{3, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{3, true}, {1, true}}, 1 },
  { {{3, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{3, true}, {1, true}, {1, true}}, 1 },
  { {{3, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{3, true}, {2, false}, {1, true}}, 2 },
  { {{3, true}, {2, true}}, 1 },
  { {{3, true}, {2, true}, {1, true}}, 1 },
  { {{3, true}, {3, true}}, 1 },
  { {{4, false}, {1, true}}, 6 },
  { {{4, false}, {1, true}, {1, false}, {1, true}}, 4 },
  { {{4, false}, {1, true}, {1, true}}, 5 },
  { {{4, false}, {1, true}, {1, true}, {1, true}}, 4 },
  { {{4, false}, {1, true}, {2, true}}, 1 },
  { {{4, false}, {2, true}}, 5 },
  { {{4, false}, {2, true}, {1, true}}, 5 },
  { {{4, false}, {3, true}}, 5 },
  { {{4, true}}, 3 },
  { {{4, true}, {1, false}, {1, true}}, 1 },
  { {{4, true}, {1, false}, {1, true}, {1, true}}, 1 },
  { {{4, true}, {1, true}}, 1 },
  { {{4, true}, {1, true}, {1, false}, {1, true}}, 1 },
  { {{4, true}, {1, true}, {1, true}}, 1 },
  { {{4, true}, {1, true}, {1, true}, {1, true}}, 1 },
  { {{4, true}, {2, false}, {1, true}}, 2 },
  { {{4, true}, {2, true}}, 1 },
  { {{4, true}, {2, true}, {1, true}}, 1 },
  { {{4, true}, {3, true}}, 1 },
  { {{5, false}, {1, true}}, 6 },
  { {{5, false}, {1, true}, {1, true}}, 4 },
  { {{5, false}, {2, true}}, 1 },
  { {{6, false}, {1, true}}, 9 },
  { {{7, false}}, 19 },
  { {{7, true}}, 1 }
};


/**
 * @brief Looks up the precomputed number of reachable full configurations for a given partial state.
 *
 * Normalizes the input state (merges adjacent unallocated, trims trailing unallocated)
 * before looking it up in the hardcoded cache.
 *
 * @param partial_state The current state of GPU partitions (allocated and unallocated).
 * @return The precomputed number of full configurations reachable from the normalized state,
 *         or 0 if the state is not found in the cache (should not happen for valid derived states).
 */
int get_precomputed_reachability_count(const std::vector<MemoryPartition>& partial_state) {
    std::vector<MemoryPartition> normalized = normalize_state(partial_state);
    auto it = precomputed_reachability_cache.find(normalized);
    if (it != precomputed_reachability_cache.end()) {
        return it->second;
    } else {
        // This case should ideally not be reached if the input state
        // is one derivable from the full_configurations used to build the cache.
        // Consider logging an error or returning a specific error code if needed.
        return 0; // State not found in the precomputed cache
    }
}


/**
 * @brief Determines a potential placement index for a new GPU partition that maximizes
 *        the number of reachable full configurations according to the precomputed cache.
 *
 * @param currentState The current state of the GPU memory partitions.
 * @param requested_size_in_slices The size of the new partition requested, in slices.
 * @return The starting index (0-based) of the slice where the new partition
 *         should be placed to maximize future partitioning options. 
 *         Returns -1 if no suitable placement is possible or the request is invalid.
 */
int find_partition_index(
    const std::vector<MemoryPartition>& currentState,
    int requested_size_in_slices)
{
    // Basic validation
    if (requested_size_in_slices <= 0 || requested_size_in_slices > A100_GPC_SLICE_NUM) {
        return -1; 
    }

    // Normalize the input state to merge adjacent unallocated blocks
    std::vector<MemoryPartition> normalized_current_state = normalize_state(currentState);
    int remaining_slices = A100_GPC_SLICE_NUM - std::accumulate(normalized_current_state.begin(), normalized_current_state.end(), 0, [](int sum, const MemoryPartition& p) { return sum + p.size_in_slices; });
    if(remaining_slices > 0){
        normalized_current_state.push_back({remaining_slices, false});
    }

    int best_index = -1;
    int max_reachable_count = -1; // Use -1 to ensure any valid count is higher

    int partition_start_slice_index = 0;
    // Iterate through the *normalized* state
    for (size_t i = 0; i < normalized_current_state.size(); ++i) {
        const MemoryPartition& current_partition = normalized_current_state[i];

        if (!current_partition.allocated && current_partition.size_in_slices >= requested_size_in_slices) {
            // This unallocated block is large enough. Iterate through possible start positions within it.
            for (int j = 0; j <= current_partition.size_in_slices - requested_size_in_slices; ++j) {
                int potential_placement_index = partition_start_slice_index + j;

                // Construct the hypothetical next state for placing at this offset j
                std::vector<MemoryPartition> next_state;
                // 1. Add partitions before the current one
                next_state.insert(next_state.end(), normalized_current_state.begin(), normalized_current_state.begin() + i);

                // 2. Add initial unallocated part (if placing with offset j > 0)
                if (j > 0) {
                    next_state.push_back({j, false});
                }

                // 3. Add the new allocated partition
                next_state.push_back({requested_size_in_slices, true});

                // 4. Add trailing unallocated part (if any remains after placing)
                int remaining_after = current_partition.size_in_slices - requested_size_in_slices - j;
                if (remaining_after > 0) {
                    next_state.push_back({remaining_after, false});
                }

                // 5. Add partitions after the current one
                if (i + 1 < normalized_current_state.size()) {
                    next_state.insert(next_state.end(), normalized_current_state.begin() + i + 1, normalized_current_state.end());
                }

                // 6. Evaluate the next state (it will be normalized again inside the lookup function)
                int reachable_count = get_precomputed_reachability_count(next_state);

                // Check if this placement results in a valid state (count > 0)
                // and if it's better than the current best valid state found.
                if (reachable_count > 0 && reachable_count > max_reachable_count) {
                    max_reachable_count = reachable_count;
                    best_index = potential_placement_index;
                }
            } // End inner loop (j) for offsets within the unallocated block
        }

        // Move to the start of the next partition
        partition_start_slice_index += current_partition.size_in_slices;
    }

    return best_index;
}



#endif // GPU_PARTITIONER_H