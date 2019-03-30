/*template <class T, size_t W, size_t H>
class Array2D
{
public:
	const int width = W;
	const int height = H;
	typedef typename T type;

	Array2D()
		: buffer(width*height)
	{
	}

	inline type& at(unsigned int x, unsigned int y)
	{
		return buffer[y*width + x];
	}

	inline const type& at(unsigned int x, unsigned int y) const
	{
		return buffer[y*width + x];
	}

private:
	std::vector<T> buffer;
}; 

void foo()
{
	Array2D<int, 800, 800> zbuffer;

	// Do something with zbuffer...
}
*/