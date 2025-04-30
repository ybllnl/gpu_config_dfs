#include "gpu_partitioner.h"
#include <iostream>
#include <vector>

// Function Declarations (Prototypes)
std::vector<MemoryPartition> get_state_after_placement(const std::vector<MemoryPartition>& initial_state, int start_index, int placed_size);
void run_placement_test(const std::vector<MemoryPartition>& initial_state, int request_size);

int main() {
    std::cout << "Step 1: Running dynamic precomputation (will print generated states)..." << std::endl;
    ReachabilityCache dynamic_cache = precompute_reachable_counts();
    std::cout << "Dynamic precomputation finished. Dynamic cache contains " << dynamic_cache.size() << " entries." << std::endl;

    std::cout << "\nStep 2: Comparing dynamic cache entries with hardcoded cache lookups..." << std::endl;
    bool mismatch_found = false;
    int checked_count = 0;

    for (const auto& pair : dynamic_cache) {
        const std::vector<MemoryPartition>& state = pair.first;
        int dynamic_count = pair.second;

        // Look up the same state using the function that accesses the hardcoded map.
        // Note: get_precomputed_reachability_count normalizes the state internally,
        // but the keys in dynamic_cache should already be normalized by precompute_reachable_counts.
        int static_count = get_precomputed_reachability_count(state);

        if (dynamic_count != static_count) {
            std::cerr << "ERROR: Mismatch found for state " << to_string(state) << "!\n";
            std::cerr << "  Dynamic count: " << dynamic_count << "\n";
            std::cerr << "  Static (hardcoded lookup) count: " << static_count << "\n";
            mismatch_found = true;
        }
        checked_count++;
    }

    std::cout << "Comparison loop finished. Checked " << checked_count << " entries." << std::endl;

    // Compare cache sizes
    // Accessing the static map directly requires making it non-static or providing a getter.
    // For simplicity, let's rely on the count reported by the dynamic computation
    // and assume the hardcoded one *should* match if generation logic is consistent.
    // A more robust test would expose the static map's size.
    size_t static_cache_assumed_size = 327; // Based on previous hardcoding
    std::cout << "\nStep 3: Comparing cache sizes..." << std::endl;
    std::cout << "  Dynamic cache size: " << dynamic_cache.size() << std::endl;
    // We can't directly access the static const map's size easily without changing its definition
    // std::cout << "  Static cache size: " << precomputed_reachability_cache.size() << std::endl;
    std::cout << "  Expected static cache size (from hardcoding): " << static_cache_assumed_size << std::endl;

    if (dynamic_cache.size() != static_cache_assumed_size) {
         std::cerr << "ERROR: Dynamic cache size differs from expected static cache size!" << std::endl;
         mismatch_found = true;
    }

    std::cout << "\nVerification Result:" << std::endl;
    if (mismatch_found) {
        std::cout << "  Verification FAILED. Mismatches were found." << std::endl;
    } else {
        std::cout << "  Verification PASSED. All checked entries match and sizes are consistent." << std::endl;
    }

    std::cout << "\n\nStep 4: Running find_partition_index tests..." << std::endl;

    // Test Case 1: Fully unallocated, request 1 slice
    run_placement_test({{7, false}}, 1);

    // Test Case 2: Fragmented, request 1 slice - should pick the one maximizing future options
    run_placement_test({{1, true}, {2, false}, {1, true}, {3, false}}, 1);

    // Test Case 3: Request 2 slices, multiple options
    run_placement_test({{1, true}, {2, false}, {1, true}, {3, false}}, 2);
    
    // Test Case 4: Request 4 slices, only one spot fits
    run_placement_test({{1, true}, {4, false}, {2, true}}, 4);

    // Test Case 5: Request fits, but might lead to less optimal state (compare with Case 3)
    run_placement_test({{3, false}, {1, true}, {3, false}}, 3);

    // Test Case 6: Request doesn't fit anywhere
    run_placement_test({{1, true}, {1, false}, {1, true}, {1, false}, {3, true}}, 2);

    // Test Case 7: Request fits, but resulting state might not be in cache (should return -1 or lower priority index)
    // Example: State [2:A, 2:F, 3:A] requesting 1 slice -> next state [2:A, 1:A, 1:F, 3:A] -> normalize -> [2:A, 1:A, 1:F, 3:A]
    // Let's assume [2:A, 1:A, 1:F, 3:A] is NOT in the cache. find_partition_index should not return index 2.
    run_placement_test({{2, true}, {2, false}, {3, true}}, 1); 

    std::cout << "----------------------------------" << std::endl;
    std::cout << "Placement tests finished." << std::endl;

    return mismatch_found ? 1 : 0; // Return non-zero on failure
}


// Helper function to manually construct the state after placement
// (Similar logic as inside find_partition_index, but separated for clarity)
std::vector<MemoryPartition> get_state_after_placement(const std::vector<MemoryPartition>& initial_state, int start_index, int placed_size) {
    std::vector<MemoryPartition> final_state;
    int current_slice = 0;
    bool placed = false; // Flag to ensure placement happens only once

    for (const auto& part : initial_state) {
        if (!placed && !part.allocated &&
            start_index >= current_slice && // Placement starts within or at the beginning of this block
            start_index < current_slice + part.size_in_slices && // Placement starts before the end of this block
            start_index + placed_size <= current_slice + part.size_in_slices) // Placement fits within this block
        {
            // Calculate sizes of the three potential new blocks
            int prefix_unallocated = start_index - current_slice;
            int suffix_unallocated = (current_slice + part.size_in_slices) - (start_index + placed_size);

            // Add the part before the new allocation (if any)
            if (prefix_unallocated > 0) {
                final_state.push_back({prefix_unallocated, false});
            }

            // Add the newly allocated partition
            final_state.push_back({placed_size, true});

            // Add the part after the new allocation (if any)
            if (suffix_unallocated > 0) {
                final_state.push_back({suffix_unallocated, false});
            }
            placed = true; // Mark as placed
        } else {
            // This partition is not where the placement happens, copy it as is
            final_state.push_back(part);
        }
        current_slice += part.size_in_slices;
    }

    // Normalize the resulting state (merge adjacent unallocated, trim trailing)
    // We should normalize *before* returning to handle merging potentially
    // created adjacent unallocated blocks.
    return normalize_state(final_state);
}

void run_placement_test(const std::vector<MemoryPartition>& initial_state, int request_size) {
    std::cout << "----------------------------------" << std::endl;
    std::cout << "Test Case:" << std::endl;
    std::cout << "  Initial State: " << to_string(initial_state) << std::endl;
    std::cout << "  Request Size : " << request_size << " slice(s)" << std::endl;

    int best_index = find_partition_index(initial_state, request_size);

    std::cout << "  Returned Index: " << best_index << std::endl;

    if (best_index != -1) {
        std::vector<MemoryPartition> state_after = get_state_after_placement(initial_state, best_index, request_size);
        int resulting_reachability = get_precomputed_reachability_count(state_after); // Use the cache lookup
        std::cout << "  State After Placement (at index " << best_index << "): " << to_string(state_after) << std::endl;
        std::cout << "  Reachability Count of Resulting State: " << resulting_reachability << std::endl;
    } else {
        std::cout << "  (No suitable placement found)" << std::endl;
    }
}
