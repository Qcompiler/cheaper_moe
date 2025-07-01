    using paramsA_float = cutlass::transform::threadblock::PredicatedTileAccessIterator
    <cutlass::MatrixShape<128, 64>, cutlass::half_t, cutlass::layout::RowMajor, 1, 
    cutlass::transform::PitchLinearWarpRakedThreadMap<cutlass::PitchLinearShape<64, 128>, 256, 
    cutlass::PitchLinearShape<8, 4>, 8>, 
    <cutlass::half_t, 8, false>, false, cutlass::layout::NoPermute>::Params;

    using paramsB_float = cutlass::transform::threadblock::PredicatedTileAccessIterator
    <cutlass::MatrixShape<64, 256>, cutlass::half_t, cutlass::layout::ColumnMajor, 0,
     cutlass::transform::PitchLinearWarpRakedThreadMap<cutlass::PitchLinearShape<64, 256>, 256, 
    cutlass::PitchLinearShape<8, 4>, 8>, 
    cutlass::Array<cutlass::half_t, 8, false>, false, cutlass::layout::NoPermute>::Params;

    // paramsA_float params_A;
    // paramsB_float params_B;


    cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::GemmShape<128, 256, 64>, 
    cutlass::transform::threadblock::PredicatedTileAccessIterator<cutlass::MatrixShape<128, 64>, 
    cutlass::half_t, cutlass::layout::RowMajor, 1, 
    cutlass::transform::PitchLinearWarpRakedThreadMap<cutlass::PitchLinearShape<64, 128>, 
    256, cutlass::PitchLinearShape<8, 4>, 8>,
     cutlass::Array<cutlass::half_t, 8, false>, false, cutlass::layout::NoPermute>, 
     cutlass::transform::threadblock::RegularTileAccessIterator<cutlass::MatrixShape<128, 64>,
      cutlass::half_t, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 64>, 0, 
      cutlass::transform::PitchLinearWarpRakedThreadMap<cutlass::PitchLinearShape<64, 128>, 
      256, cutlass::PitchLinearShape<8, 4>, 8>, 16>, cutlass::arch::CacheOperation::Global, 
      cutlass::transform::threadblock::PredicatedTileAccessIterator<cutlass::MatrixShape<64, 256>,
       cutlass::half_t, cutlass::layout::ColumnMajor, 0, 
       cutlass::transform::PitchLinearWarpRakedThreadMap<cutlass::PitchLinearShape<64, 256>, 256, 
       cutlass::PitchLinearShape<8, 4>, 8>, cutlass::Array<cutlass::half_t, 8, false>, false, 
       cutlass::layout::NoPermute>, cutlass::transform::threadblock::RegularTileAccessIterator<cutlass::MatrixShape<64, 256>,
        cutlass::half_t, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16, 64>, 1,
         cutlass::transform::PitchLinearWarpRakedThreadMap<cutlass::PitchLinearShape<64, 256>, 256, 
         cutlass::PitchLinearShape<8, 4>, 8>, 16>, cutlass::arch::CacheOperation::Global, float, cutlass::layout::RowMajor, 
         cutlass::gemm::threadblock::MmaPolicy<cutlass::gemm::warp::MmaTensorOp<cutlass::gemm::GemmShape<64, 64, 64>, 
         cutlass::half_t, cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 64>, 
         cutlass::half_t, cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16, 64>, 
         float, cutlass::layout::RowMajor, cutlass::gemm::warp::MmaTensorOpPolicy<cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 8, 16>, 32, c
         utlass::half_t, cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::ColumnMajor, 
         float, cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>, cutlass::MatrixShape<1, 1>>, 1, false, __nv_bool>, 
         cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>, 1>, 3, 
    cutlass::gemm::SharedMemoryClearOption::kNone, __nv_bool>




    argument types are: (cutlass::epilogue::thread::symmetric::LinearCombinationDequant
    <cutlass::half_t, 8, int32_t, cutlass::half_t, cutlass::epilogue::thread::symmetric::MyScaleType::Dequantize, 
    cutlass::FloatRoundStyle::round_to_nearest, cutlass::half_t>, 
    cutlass::epilogue::threadblock::PredicatedTileIterator<cutlass::epilogue::threadblock::OutputTileOptimalThreadMap
    <cutlass::epilogue::threadblock::OutputTileShape<256, 8, 2, 1, 1>, 
    cutlass::epilogue::threadblock::OutputTileShape<1, 8, 1, 1, 8>, 256, 8, 16>,
     cutlass::half_t, false, cutlass::layout::NoPermute, false>,
      cutlass::Array<int32_t, 128, true>, 
      cutlass::Array<float, 128, true>, cutlass::epilogue::threadblock::PredicatedTileIterator<cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<cutlass::epilogue::threadblock::OutputTileShape<256, 8, 2, 1, 1>, cutlass::epilogue::threadblock::OutputTileShape<1, 8, 1, 1, 8>, 256, 8, 16>, cutlass::half_t, false, cutlass::layout::NoPermute, false>, cutlass::epilogue::threadblock::symmetric::PredicatedVRowIterator<cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<cutlass::epilogue::threadblock::OutputTileShape<256, 8, 2, 1, 1>, cutlass::epilogue::threadblock::OutputTileShape<1, 8, 1, 1, 8>, 256, 8, 16>, cutlass::half_t, false, cutlass::layout::NoPermute, false>, cutlass::epilogue::threadblock::symmetric::PredicatedVColIterator<cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<cutlass::epilogue::threadblock::OutputTileShape<256, 8, 2, 1, 1>, cutlass::epilogue::threadblock::OutputTileShape<1, 8, 1, 1, 8>, 256, 8, 16>, cutlass::half_t, false, cutlass::layout::NoPermute, false>)