/// Automatically calculates the window and level for this image frame.
//bool AutoWindowLevel(double &Window, double &Level, int Frame,
//    double Intercept=0.0, double Slope=1.0, BOOL HasPadding=FALSE, int PaddingValue=0) ;

namespace std::string

// assume data is int* (for now)
void AutoWindowLevel(double &Window, double &Level, int *data, int Frame,
    double Intercept, double Slope, bool HasPadding, int PaddingValue, std::String BodyPartExamined)
{
	if (m_dataType > AShort)	// Currently can only handle 16 bit data or less.
		return false;

	unsigned long   *cumul_histo;
	UINT            num_bins,
					num_pixels,
					Number;
	int high, low;
	int MAX_GAP = 1000;
	int MAX_VAL = -1;
	int MIN_VAL = -1;
	bool UseMaxMin = false;

	num_bins = 0;
	Number = 2;

	if (m_pDicomObject)
	{
		std::string bodyPartExamined;
		if (m_pDicomObject->GetValue(0x0018, 0x0015, bodyPartExamined) == ADO_OK)	// BodyPartExamined
			if (bodyPartExamined.find("SPINE") != std::string::npos) // && ! v->coil_filtered[slice] )
			{
				Number = 10;

				std::string modality;
				if (m_pDicomObject->GetValue(0x0008, 0x0060, modality) == ADO_OK)	// Modality
				{
					unsigned short bitsAllocated = 0;
					if (m_pDicomObject->GetValue(0x0028, 0x0100, bitsAllocated) == ADO_OK)	// BitsAllocated
					{
						if ((bitsAllocated == 16) && (modality == "MR"))
						{
							UseMaxMin = true;
							MIN_VAL = -1000;
							MAX_VAL = 500;
						}
					}
				}
			}
	}

	num_pixels = m_width * m_height;  // need to pass in width and height
	high = low = *data;

	// cumul_histo needs to be a long pointer because the range
	// of a short (65536) is too small for a 256x256 flat image or
	// images of larger dimensions
	cumul_histo = new ULONG[0x10000];  // 65536, big enough to hold all 16-bit values
	memset(cumul_histo, 0, 0x10000 * sizeof(long));

	// WJR - 07/30/99
	//   Convert all values to USHORT for indexing the histogram array
	USHORT usValue;                     // Will give us our offset in the cumul_histo array
	USHORT usPad = (USHORT) PaddingValue;  // Do the typecast just once
	for ( UINT i = 0; i < num_pixels; i++)
	{
		usValue = (USHORT) *(data + i);
		if ( HasPadding == FALSE || usValue != usPad )
		{
			if ( UseMaxMin )
			{
				if ( *(data + i) < (T) MIN_VAL || *(data + i) > (T) MAX_VAL )
				{
					continue;
				}
			}

			// Count the number of bins that pass "Number" pixels
			if ( cumul_histo[usValue] == Number )
			{
				num_bins++;
			}

			++cumul_histo[usValue];
		}
	}

	// If we're signed, we'll have to offset our index later
	T negvalue = -1;
	BOOL Signed = FALSE;
	if ( negvalue < 0 )
	{
		Signed = TRUE;
	}

	// Instead of testing to Min and Max for all pixels in the image, just
	//   loop through all of the histogram entries
	int prev_bin = ((Signed)?(0-0xFFFF):0);
	int iValue;
	int Bin_num = 0;          // Long value to add up all the bins
	USHORT valid_bins = 0; // Number of bins meeting our criteria
	USHORT OFFSET = ((Signed)?0x8000:0x0000);  // If signed data, count vals above 0x8000 first
	USHORT limit = 0xFFFF;
	USHORT index = 0;

	do
	{
		usValue = index + OFFSET;  // If signed values, get the negative values first

		iValue = ((Signed)?(int)(SHORT)usValue:(int)(USHORT)usValue);

		// Ignore stray pixel values that may be used as padding or overlay text
		if ( cumul_histo[usValue] > (ULONG) Number )
		{
			// Check for a big gap in the histogram
			if ( iValue - prev_bin > MAX_GAP && valid_bins > 0 )
			{
				if ( ((UINT(valid_bins) * 100) / num_bins) < 10 )  // Less than 10 percent of the colors affected
				{
					// Big Gap, ignore the other value
					valid_bins = 0;
					Bin_num = 0;
					low = (T) iValue;
				}
				else
				//else if ( ( ((num_bins - valid_bins) * 100) / num_bins) < 10 )
				{
					break;  // Get out now!
				}
			}

			// Check for the lowest bin
			if ( valid_bins == 0 )
			{
				low = (T) iValue;
			}

			// Increment the values
			valid_bins++;
			Bin_num += iValue;

			// Make sure to set prev_bin
			prev_bin = iValue;
			high = (T) iValue;
		}

		// Increment the counter
		index++;
	} while ( index != 0x0000 );  // Will stop on the wrap around

	if ( valid_bins < 1 )
	{
		valid_bins = 1;  // This avoids a divide by zero error
	}


	// Try to brighten up the images a bit...
	UINT LevelAdjust = 0;	//(high - low) / 20;

	Level = (int((Bin_num/valid_bins) * Slope) + Intercept) - LevelAdjust;
	Window = int((high - low) * Slope);

	if ( Window <= 0 )
		Window = high - low;

	delete [] cumul_histo;
}