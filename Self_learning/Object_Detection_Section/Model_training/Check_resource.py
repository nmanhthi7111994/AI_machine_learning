import time
import sys

# Define the compute unit usage rate (units per hour)
usage_rate_per_hour = 0.07

# Convert to units per second
usage_rate_per_second = usage_rate_per_hour / 3600

# Set the maximum compute units you want to use (adjust this value)
max_compute_units = 0.0003  # Example value, set to the actual available units

# Function to calculate compute units used
def calculate_compute_units_used(start_time, usage_rate_per_second):
    elapsed_time = time.time() - start_time
    compute_units_used = elapsed_time * usage_rate_per_second
    return compute_units_used

# Start time of the session
start_time = time.time()

try:
    while True:
        # Calculate compute units used so far
        compute_units_used = calculate_compute_units_used(start_time, usage_rate_per_second)
        
        # Print the result
        print(f"Approximate Compute Units used so far: {compute_units_used:.6f} units")
        
        # Check if compute units used have reached the maximum threshold
        if compute_units_used >= max_compute_units:
            print("Maximum compute units reached. Stopping the process.")
            sys.exit()  # Stop the process
        
        # Wait for 10 seconds before checking again
        time.sleep(10)
        
except KeyboardInterrupt:
    print("Monitoring stopped.")
