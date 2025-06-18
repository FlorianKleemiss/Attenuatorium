"""
Attenuatorium: A Crystallographic Reflection Data Analysis Tool

This module provides functionality for handling, analyzing, and visualizing 
crystallographic reflection data from HKL files. It enables users to manage 
intensity thresholds, merge datasets, and perform other crystallographic data 
operations.

Core components:
- reflection: Class representing a single reflection with HKL indices, intensity, and sigma
- cell: Class representing unit cell parameters and related calculations
- reflection_list: Class handling collections of reflections with methods for 
  analysis and manipulation

The module supports reading HKL files, finding common reflections between datasets,
calculating intensities, and other operations required for crystallographic data processing.
"""

import math
import numpy as np
import sys

class reflection():
    """
    Represents a single crystallographic reflection with its indices, intensity, and sigma.
    
    This class encapsulates the data for a single reflection in a diffraction pattern,
    identified by its Miller indices (h,k,l), intensity value, and sigma (error estimate).
    """
    def __init__(self) -> None:
        self.ind = (0,0,0)
        self.intensity = 0.0
        self.sigma = 0.0
    def __init__(self, hkl, i, s) -> None:
        self.ind = hkl
        self.intensity = i
        self.sigma = s
    def l(self) -> int:
        return self.ind[2]
    def k(self) -> int:
        return self.ind[1]
    def h(self) -> int:
        return self.ind[0]
    def i(self) -> float:
        return self.intensity
    def s(self) -> float:
        return self.sigma
    def index(self) -> tuple:
        return self.ind
    def i_over_s(self) -> float:
        return self.intensity / self.sigma
    def ratio(self, compare) -> float:
        return self.i() / compare.i()
    def sigma_ratio(self, compare) -> float:
        return self.s() / compare.s()

class cell():
    """
    Represents a crystallographic unit cell with its parameters and related calculations.
    
    This class stores unit cell parameters (a, b, c, alpha, beta, gamma) and wavelength,
    and provides methods to calculate d-spacing and other crystallographic properties
    based on these parameters.
    """
    def __init__(self) -> None:
        self.a = self.b = self.c = 5
        self.alpha = self.beta = self.gamma = 90
        self.ca = self.cb = self.cg = 1.0
        self.sa = self.sb = self.sg = 0.0
        self.wavelength = 0.71
    def __init__(self,a,b,c,alpha,beta,gamma,wl) -> None:
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.gamma = float(gamma)
        self.wavelength = float(wl)
        self.ca = math.cos(math.pi/180.0*self.alpha)
        self.cb = math.cos(math.pi/180.0*self.beta)
        self.cg = math.cos(math.pi/180.0*self.gamma)
        self.sa = math.sin(math.pi/180.0*self.alpha)
        self.sb = math.sin(math.pi/180.0*self.beta)
        self.sg = math.sin(math.pi/180.0*self.gamma)

    def get_d_of_hkl(self, hkl) -> float:
        # Handle [0,0,0] as a special case
        if hkl[0] == 0 and hkl[1] == 0 and hkl[2] == 0:
            return 999.9  # Return a large value for d when hkl = [0,0,0]

        upper = pow(hkl[0], 2) * pow(self.sa, 2) / pow(self.a, 2) \
              + pow(hkl[1], 2) * pow(self.sb, 2) / pow(self.b, 2) \
              + pow(hkl[2], 2) * pow(self.sg, 2) / pow(self.c, 2) \
              + 2.0 * hkl[1] * hkl[2] / (self.b * self.c) * (self.cb * self.cg - self.ca) \
              + 2.0 * hkl[0] * hkl[2] / (self.a * self.c) * (self.cg * self.ca - self.cb) \
              + 2.0 * hkl[0] * hkl[1] / (self.a * self.b) * (self.ca * self.cb - self.cg)

        lower = 1 - pow(self.ca, 2) - pow(self.cb, 2) - pow(self.cg, 2) + 2 * self.ca * self.cb * self.cg

        # Prevent division by zero
        if upper == 0 or lower == 0:
            return 999.9

        d = np.sqrt(lower / upper)
        return d

    def get_stl_of_hkl(self,hkl) -> float:
        return 1.0 / (2 * self.get_d_of_hkl(hkl))

class reflection_list():
    """
    Manages a collection of crystallographic reflections with analysis capabilities.
    
    This class provides functionality to store, retrieve, and analyze collections of
    reflections. It supports operations like finding common reflections between datasets,
    calculating average intensities, and other crystallographic data processing tasks.
    
    The internal storage uses efficient NumPy arrays after data collection is complete,
    providing both memory efficiency and fast operations.
    """
    def __init__(self) -> None:
        # Store HKLs, intensities, and sigmas in separate lists initially
        # self.refl_list = np.array([[],[],[]]) # Old structure
        self._hkl_tuples = []  # Store (h,k,l) tuples
        self._intensities = [] # Store intensities
        self._sigmas = []      # Store sigmas
        self._data_finalized = False # Flag to indicate if data is converted to NumPy arrays

        self.min_d = 0.0
        self.max_d = 0.0
        self.name = ""
        self.cell = None # Initialize cell attribute
        self.refl_list_np = None # This will hold the final NumPy array structure

    def set_cell(self, a, b, c, alpha, beta, gamma, wavelength):
        """
        Sets the crystallographic unit cell parameters for this reflection list.
        
        Args:
            a, b, c: Unit cell dimensions in Angstroms
            alpha, beta, gamma: Unit cell angles in degrees
            wavelength: X-ray wavelength in Angstroms
        """
        self.cell = cell(a, b, c, alpha, beta, gamma, wavelength)

    def _finalize_data_structure(self):
        """
        Converts internal data from Python lists to NumPy arrays for efficient operations.
        
        This method should be called once after all data has been appended to the reflection_list,
        typically after reading all reflections from a file. It converts the internal 
        Python lists to more efficient NumPy arrays for subsequent operations.
        """
        # Convert lists to NumPy arrays for efficient operations
        # This should be called once after all appends are done (e.g., at the end of read_hkl)
        if not self._data_finalized:
            if self._hkl_tuples: # Check if there's any data
                # self.refl_list_np[0] will be a 2D NumPy array where each row is an HKL [h,k,l]
                # This happens because self._hkl_tuples is a list of numeric tuples.
                self.refl_list_np = [
                    np.array(self._hkl_tuples, dtype=int), # Store as int array for unique_hkl compatibility with axis=0
                    np.array(self._intensities, dtype=float),
                    np.array(self._sigmas, dtype=float)
                ]
            else:
                self.refl_list_np = [
                    np.array([], dtype=int).reshape(0,3), # Ensure it's a 2D array even when empty
                    np.array([], dtype=float),
                    np.array([], dtype=float)
                ]
            self._data_finalized = True

    def append(self, h, k, l, i, s) -> None:
        if self._data_finalized:
            raise Exception("Cannot append after data structure is finalized.")

        hkl_tuple_to_append = (int(h),int(k),int(l)) # Ensure integers
        self._hkl_tuples.append(hkl_tuple_to_append)
        self._intensities.append(float(i))
        self._sigmas.append(float(s))

    def index_tuple(self) -> np.ndarray: # Returns array of HKL tuples
        if not self._data_finalized: self._finalize_data_structure()
        return self.refl_list_np[0]

    def unique_hkl(self) -> np.ndarray: # Returns unique HKLs as a 2D NumPy array (N_unique_hkls, 3)
        if not self._data_finalized: self._finalize_data_structure()
        if self.refl_list_np[0].size == 0:
            return np.array([], dtype=int).reshape(0,3) # Return empty 2D array

        # Use axis=0 to find unique rows (HKLs)
        # self.refl_list_np[0] is a 2D array of [h,k,l] rows
        result = np.unique(self.refl_list_np[0], axis=0)
        return result

    def get_hkl(self,hkl_target) -> np.ndarray:
        # hkl_target is expected to be a tuple (h,k,l) or something convertible to it.
        if not self._data_finalized: self._finalize_data_structure()

        processed_hkl_target_array = None
        if isinstance(hkl_target, np.ndarray) and hkl_target.shape == (3,):
            try:
                processed_hkl_target_array = hkl_target.astype(int)
            except (ValueError, TypeError):
                pass
        elif isinstance(hkl_target, (list, tuple)) and len(hkl_target) == 3:
            try:
                processed_hkl_target_array = np.array(hkl_target, dtype=int)
            except (ValueError, TypeError):
                pass

        if processed_hkl_target_array is None:
            return np.array([[],[],[]], dtype=object) # Return empty structure if target is not a valid HKL

        # self.refl_list_np[0] is a 2D NumPy array of HKLs (Nx3)
        all_hkls_np_array = self.refl_list_np[0]

        if all_hkls_np_array.size == 0:
            return np.array([[],[],[]], dtype=object)

        # Perform row-wise comparison
        # This creates a boolean array where each element is True if the row in all_hkls_np_array matches processed_hkl_target_array
        condition = np.all(all_hkls_np_array == processed_hkl_target_array, axis=1)
        matching_indices = np.where(condition)[0]

        if matching_indices.size == 0:
            return np.array([[],[],[]], dtype=object)

        # Return a 3xN array for the matching reflections
        # For consistency with previous structure, we return HKLs as object array of tuples/lists if needed by GUI
        # but internal operations benefit from direct NumPy arrays.
        # For now, let's return the raw data slices, which will be numeric arrays.
        # The GUI part that consumes this might need adjustment if it strictly expects tuples in the first part.
        # However, get_average_intensities_for_unique_hkls will now work correctly with this.

        # Extracting HKLs for the matches: these will be rows from self.refl_list_np[0]
        matched_hkls = self.refl_list_np[0][matching_indices] # This is a 2D array of matched HKLs

        # To maintain the [hkl_tuples_array, intensities_array, sigmas_array] structure:
        return np.array([
            [tuple(hkl) for hkl in matched_hkls], # Convert matched HKL arrays back to list of tuples for the output structure
            self.refl_list_np[1][matching_indices],
            self.refl_list_np[2][matching_indices]
        ], dtype=object)

    def has_hkl(self,hkl_target) -> bool: # hkl_target should be a tuple
        if not self._data_finalized: 
            self._finalize_data_structure()
        hkl_target_tuple = tuple(hkl_target.tolist()) if isinstance(hkl_target, np.ndarray) else tuple(hkl_target)
        return hkl_target_tuple in self.refl_list_np[0] # Efficient if self.refl_list_np[0] is a set, but it's an array.
                                                      # For arrays, this still involves a scan.

    def get_unique_hkl_data(self):
        """Processes reflection data to find unique HKLs and their averaged intensities and sigmas."""
        if not self._data_finalized: self._finalize_data_structure()

        all_hkls = self.refl_list_np[0]
        all_intensities = self.refl_list_np[1]
        all_sigmas = self.refl_list_np[2]

        if all_hkls.size == 0:
            return np.array([], dtype=int).reshape(0,3), np.array([]), np.array([])

        # Get unique HKLs, the inverse map, and counts for averaging
        unique_hkls_array, inverse_indices, counts = np.unique(
            all_hkls, axis=0, return_inverse=True, return_counts=True
        )

        # Calculate sum of intensities and sigmas for each unique HKL
        sum_intensities = np.bincount(inverse_indices, weights=all_intensities)
        sum_sigmas = np.bincount(inverse_indices, weights=all_sigmas)

        # Calculate average intensities and sigmas
        avg_intensities = sum_intensities / counts
        avg_sigmas = sum_sigmas / counts

        return unique_hkls_array, avg_intensities, avg_sigmas

    def get_common_reflections(self, other_list):
        """
        Finds common HKLs between this list and another, returning averaged data for them.
        
        This method compares two reflection_list objects and identifies reflections with
        the same HKL indices in both datasets. For these common reflections, it returns
        the HKL indices along with the averaged intensities and sigmas from both datasets.
        
        Args:
            other_list: Another reflection_list object to compare with this one
            
        Returns:
            tuple: (common_hkls, self_intensities, self_sigmas, other_intensities, other_sigmas)
                  All return values are NumPy arrays containing data for the common reflections
                  
        Raises:
            TypeError: If other_list is not a reflection_list object
        """
        if not isinstance(other_list, reflection_list):
            raise TypeError("Argument must be a reflection_list object.")

        # Get unique, averaged data for both reflection lists
        self_unique_hkls, self_avg_i, self_avg_s = self.get_unique_hkl_data()
        other_unique_hkls, other_avg_i, other_avg_s = other_list.get_unique_hkl_data()

        empty_hkl_array = np.array([], dtype=int).reshape(0,3)
        empty_float_array = np.array([])

        if self_unique_hkls.size == 0 or other_unique_hkls.size == 0:
            return empty_hkl_array, empty_float_array, empty_float_array, empty_float_array, empty_float_array

        # Create dictionaries mapping HKL tuples to their (averaged intensity, averaged sigma)
        map_self_data = {tuple(hkl): (self_avg_i[i], self_avg_s[i])
                         for i, hkl in enumerate(self_unique_hkls)}
        map_other_data = {tuple(hkl): (other_avg_i[i], other_avg_s[i])
                          for i, hkl in enumerate(other_unique_hkls)}

        # Find common HKL tuples
        common_hkl_tuples_set = set(map_self_data.keys()).intersection(map_other_data.keys())

        if not common_hkl_tuples_set:
            return empty_hkl_array, empty_float_array, empty_float_array, empty_float_array, empty_float_array

        # Sort common HKLs for consistent output order (e.g., by h, then k, then l)
        sorted_common_hkl_tuples = sorted(list(common_hkl_tuples_set))

        out_common_hkls_list = []
        out_self_i_list = []
        out_self_s_list = []
        out_other_i_list = []
        out_other_s_list = []

        for hkl_tuple in sorted_common_hkl_tuples:
            out_common_hkls_list.append(list(hkl_tuple)) # Store as list for np.array(..., dtype=int)

            s_i, s_s = map_self_data[hkl_tuple]
            out_self_i_list.append(s_i)
            out_self_s_list.append(s_s)

            o_i, o_s = map_other_data[hkl_tuple]
            out_other_i_list.append(o_i)
            out_other_s_list.append(o_s)

        return (
            np.array(out_common_hkls_list, dtype=int),
            np.array(out_self_i_list, dtype=float),
            np.array(out_self_s_list, dtype=float),
            np.array(out_other_i_list, dtype=float),
            np.array(out_other_s_list, dtype=float)
        )

# Ensure read_hkl and other functions are below the class definition if they are in this file
# For example:
# def read_hkl(filename):
# ... (rest of the file)
def read_hkl(filename) -> reflection_list:
    """
    Reads crystallographic reflection data from an HKL file.
    
    This function parses HKL format files, extracting reflection data (indices, intensities,
    sigmas) and optional unit cell parameters if present. It creates and returns a
    reflection_list object containing all the parsed data.
    
    Args:
        filename: Path to the HKL file to read
        
    Returns:
        reflection_list: Object containing the parsed reflection data
    """
    listy = reflection_list() # Initializes with internal Python lists

    lines = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return listy # Return empty list

    last_line_idx = -1 # Initialize to -1, indicating 000 line not found

    for i, line in enumerate(lines):
        if not line.strip(): # Skip empty lines
            continue
        if "   0   0   0    0.00    0.00   0" in line: # More robust check
            last_line_idx = i
            continue # Skip end marker, but do not break
        try:
            # Ensure line has enough characters before slicing
            if len(line) >= 28:
                h = int(line[0:4].strip()) # Adjusted slicing, added strip
                k = int(line[4:8].strip())
                l_val = int(line[8:12].strip()) # Renamed to avoid lint warning
                i_val = float(line[12:20].strip())
                s_val = float(line[20:28].strip())
                listy.append(h, k, l_val, i_val, s_val)
            # else: log malformed line?
        except ValueError as ve:
            # print(f"Skipping malformed line in {filename} (line {i+1}): {line} - {ve}")
            continue # Skip lines that don't conform to format

    # Finalize data structure after all appends
    listy._finalize_data_structure()

    # Process CELL line if 000 line was found and there are lines after it
    if last_line_idx != -1 and last_line_idx < len(lines) -1 : # Check if last_line_idx is valid
        for n_offset in range(last_line_idx + 1, len(lines)): # Iterate from line after 000
            line_content = lines[n_offset].strip()
            if "CELL" in line_content:
                parts = line_content.split()
                # Expecting "CELL wl a b c alpha beta gamma"
                if len(parts) == 8: # CELL + 7 values
                    try:
                        # parts[0] is "CELL"
                        wl = float(parts[1])
                        a = float(parts[2])
                        b = float(parts[3])
                        c_val = float(parts[4]) # Renamed
                        alpha = float(parts[5])
                        beta = float(parts[6])
                        gamma = float(parts[7])
                        listy.set_cell(a, b, c_val, alpha, beta, gamma, wl)
                        break # Found and processed CELL line
                    except ValueError as ve_cell:
                        # print(f"Error parsing CELL line in {filename}: {line_content} - {ve_cell}")
                        pass # Or handle error more explicitly
                # else: print(f"Malformed CELL line in {filename}: {line_content}")

    return listy

if __name__ == "__main__":
    print("This is a library, not a script. Use the GUI to run it.")
    sys.exit(0)
