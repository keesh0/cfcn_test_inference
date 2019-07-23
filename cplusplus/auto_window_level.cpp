/// Automatically calculates the window and level for this image frame.
//bool AutoWindowLevel(double &Window, double &Level, int Frame,
//    double Intercept=0.0, double Slope=1.0, BOOL HasPadding=FALSE, int PaddingValue=0) ;

using namespace std::string;

// assume data is int* (for now)
// Pass in original slope and intercept?
void AutoWindowLevel(int *data, int width, int height,
    double Intercept, double Slope, bool HasPadding, int PaddingValue,
    double &Window, double &Level)
{
	// Currently can only handle 16 bit data or less.

	unsigned long *cumul_histo = nullptr;
	unsigned int num_bins, num_pixels, Number;
	int high, low;
	int MAX_GAP = 1000;
	int MAX_VAL = -1;
	int MIN_VAL = -1;
	bool UseMaxMin = false;

	num_bins = 0;
	Number = 2;

    // body part "SPINE" and modlaity "MR" and bits allocated = 16

	num_pixels = width * height;  // need to pass in width and height
	high = low = *data;

	// cumul_histo needs to be a long pointer because the range
	// of a short (65536) is too small for a 256x256 flat image or
	// images of larger dimensions
	cumul_histo = new unsigned long[0x10000];  // 65536, big enough to hold all 16-bit values
	memset(cumul_histo, 0, 0x10000 * sizeof(unsigned long));

	// WJR - 07/30/99
	//   Convert all values to USHORT for indexing the histogram array
	unsigned short usValue;                     // Will give us our offset in the cumul_histo array
	unsigned short usPad = (unsigned short) PaddingValue;  // Do the typecast just once
	for ( uint i = 0; i < num_pixels; i++)
	{
		usValue = (unsigned short) *(data + i);
		if ( HasPadding == false || usValue != usPad )
		{
			if ( UseMaxMin )
			{
				if ( *(data + i) < (int) MIN_VAL || *(data + i) > (int) MAX_VAL )
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
	int negvalue = -1;
	bool Signed = false;
	if ( negvalue < 0 )
	{
		Signed = true;
	}

	// Instead of testing to Min and Max for all pixels in the image, just
	//   loop through all of the histogram entries
	int prev_bin = ((Signed)?(0-0xFFFF):0);
	int iValue;
	int Bin_num = 0;          // Long value to add up all the bins
	unsigned short valid_bins = 0; // Number of bins meeting our criteria
	unsigned short OFFSET = ((Signed)?0x8000:0x0000);  // If signed data, count vals above 0x8000 first
	unsigned short limit = 0xFFFF;
	unsigned short index = 0;

	do
	{
		usValue = index + OFFSET;  // If signed values, get the negative values first

		iValue = ((Signed)?(int)(short)usValue:(int)(unsigned short)usValue);

		// Ignore stray pixel values that may be used as padding or overlay text
		if ( cumul_histo[usValue] > (unsigned long) Number )
		{
			// Check for a big gap in the histogram
			if ( iValue - prev_bin > MAX_GAP && valid_bins > 0 )
			{
				if ( ((unsigned int(valid_bins) * 100) / num_bins) < 10 )  // Less than 10 percent of the colors affected
				{
					// Big Gap, ignore the other value
					valid_bins = 0;
					Bin_num = 0;
					low = (int) iValue;
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
				low = (int) iValue;
			}

			// Increment the values
			valid_bins++;
			Bin_num += iValue;

			// Make sure to set prev_bin
			prev_bin = iValue;
			high = (int) iValue;
		}

		// Increment the counter
		index++;
	} while ( index != 0x0000 );  // Will stop on the wrap around

	if ( valid_bins < 1 )
	{
		valid_bins = 1;  // This avoids a divide by zero error
	}


	// Try to brighten up the images a bit...
	uint LevelAdjust = 0;	//(high - low) / 20;

	Level = (int((Bin_num/valid_bins) * Slope) + Intercept) - LevelAdjust;
	Window = int((high - low) * Slope);

	if ( Window <= 0 )
		Window = high - low;

	delete [] cumul_histo;
}