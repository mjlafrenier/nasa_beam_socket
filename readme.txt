README

NASA Beam Socket

<------------------------Revision Control---------------------->
<---Revision---><-----Date-----><-----User-----><---Comments--->
v0.0b		2/11/2025	M. LaFrenier	Initial release

Description:
This tool analyzes a pin socket joint by providing shear and moment loads from Ref 1.
This tool is setup for the following conditions:
	- pin is a solid cross section
	- socket is a hollow tube cross section
	- pin socket contact is continuous

The following margins of safety are provided:
	Pin Shear: average pin shear, V/A
	Pin Bending: tension only, Mc/I
	Pin Shear + Bending Interaction: exponents <2,2>
	Socket Shear: average pin shear, V/A
	Socket Bending: tension only, Mc/I
	Socket Shear + Bending Interaction: exponents <2,2>
	
<1> Rash, Larry C. "Strength Evaluation of Socket Joints". NASA Contractor Report 4608. June 1994

Input:
- Input CSV
	ID: User ID 
	D_pin [in]: diameter of the pin
	Di_socket [in]: inner diameter of the socket
	t_socket [in]: thickness of the socket hollow pin idealization
	L [in]: total length per <1>
	a [in]: length of socket unused/unloaded per <1>
	b [in]: length of socket
	n: number of calculation points within socket region (L-b through L)
	SC: subcase
	Fo [lb]: applied load at x = 0
	Mo [in-lb]: applied moment at x = 0
	Fs_pin [ksi]: pin shear allowable in ksi
	Fb_pin [ksi]: pin bending allowable in ksi
	Fs_socket [ksi]: socket shear allowable in ksi
	Fb_socket [ksi]: socket bending allowable in ksi
	FF: Fitting factor; applied during MS calculation

Output:
- CSV Files:
	- beam_socket_analysis_o_MMDDYYYY.csv: Contains all beam socket analysis data
	- beam_socket_ms_summary_o_MMDDYYYY.csv: Contains beam socket MS summary by failure mode with all columns
	- beam_socket_ms_red_summary_o_MMDDYYYY.csv: Contains beam socket reduced MS summary by failure mode with less columns
- XS Image Folder (o_xs_images):
	- jpeg images of all VM diagrams; displays cw points and stress recovery points

Instructions:
1) Prerequisites
	a.) Python: Install CAE - Anaconda 2023.09
	b.) Libraries: See script
2.) Create input CSV
3.) Run batch script "RunPy.bat"
4.) Follow prompts
	a.) Select csv file.
5.) Review csv data, images

Helpful Links:


