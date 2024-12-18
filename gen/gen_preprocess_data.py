from src.data.preprocess import Preprocessor
from src.helper.constant import (
    get_spots_csv, get_edges_csv, get_xml_path, get_utids_csv, 
    get_spots_interp_csv, get_spots_vel_csv, get_mitosis_csv )

def run_preprocessing():
    # Configuration
    sequence_id = 'FLD_7'
    
    gaps = [1, 2, 4, 8, 16]
    # Get paths for the required files
    xml_path = get_xml_path(sequence_id)
    spots_csv_path = get_spots_csv(sequence_id)
    edges_csv_path = get_edges_csv(sequence_id)
    utids_csv_path = get_utids_csv(sequence_id)
    spots_interp_csv_path = get_spots_interp_csv(sequence_id)
    spots_vel_csv_path = get_spots_vel_csv(sequence_id)
    mito_csv_path = get_mitosis_csv(sequence_id)

    # Initialize Preprocessor and run the preprocessing functions
    preprocessor = Preprocessor()
    
    # Convert XML to CSV files
    #preprocessor.spots_xml_to_csv(xml_path, spots_csv_path)
    #preprocessor.edges_xml_to_csv(xml_path, edges_csv_path)

    # Find unique Track IDs and generate the UTID CSV
    preprocessor.find_unique_trkid(edges_csv_path, utids_csv_path)

    # Interpolate the spots and generate the interpolated spots CSV
    preprocessor.interpolate_spots(spots_csv_path, utids_csv_path, spots_interp_csv_path)

    # Append velocity data to the interpolated spots CSV
    preprocessor.append_velocity(spots_interp_csv_path, spots_vel_csv_path, gaps_= gaps, noise_std_=[0])

    # Identify mitosis frames and generate the mitosis CSV
    preprocessor.find_mitosis_frame(edges_csv_path, utids_csv_path, mito_csv_path)
    
    for gap in gaps:
        preprocessor.generate_prelap(sequence_id, gap)

if __name__ == '__main__':
    run_preprocessing()
