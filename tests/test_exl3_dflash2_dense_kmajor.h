#pragma once

void run_dflash2_dense_kmajor_qualification() {
    constexpr std::size_t guard=32;
    constexpr std::uint16_t canary=0x3555;
    struct Shape {int rows,k,n;};
    std::size_t values=0;
    int cases=0;
    for(const auto shape:std::array<Shape,2>{{{1,32,33},{7,5120,1280}}}) {
        const auto input_count=static_cast<std::size_t>(shape.rows)*shape.k;
        const auto weight_count=static_cast<std::size_t>(shape.n)*shape.k;
        const auto output_count=static_cast<std::size_t>(shape.rows)*shape.n;
        std::vector<std::uint16_t> input(input_count),output_major(weight_count),
            kmajor(weight_count),initial(output_count+2*guard,canary);
        std::uint32_t state=0x91e10da5U;
        auto next_half=[&]() {
            state^=state<<13;state^=state>>17;state^=state<<5;
            return static_cast<std::uint16_t>(((state>>31)<<15)|
                (((state%20)+3)<<10)|(state&0x03ffU));
        };
        for(auto& value:input)value=next_half();
        for(auto& value:output_major)value=next_half();
        for(int col=0;col<shape.n;++col)for(int i=0;i<shape.k;++i)
            kmajor[static_cast<std::size_t>(i)*shape.n+col]=
                output_major[static_cast<std::size_t>(col)*shape.k+i];
        DeviceBuffer input_device(input.size()*sizeof(input[0]));
        DeviceBuffer output_major_device(output_major.size()*sizeof(output_major[0]));
        DeviceBuffer kmajor_device(kmajor.size()*sizeof(kmajor[0]));
        DeviceBuffer control_device(initial.size()*sizeof(initial[0]));
        DeviceBuffer candidate_device(initial.size()*sizeof(initial[0]));
        cuda_check(cudaMemcpy(input_device.get(),input.data(),input_device.bytes(),cudaMemcpyHostToDevice),
            "draft dense input upload");
        cuda_check(cudaMemcpy(output_major_device.get(),output_major.data(),output_major_device.bytes(),cudaMemcpyHostToDevice),
            "draft dense output-major upload");
        cuda_check(cudaMemcpy(kmajor_device.get(),kmajor.data(),kmajor_device.bytes(),cudaMemcpyHostToDevice),
            "draft dense K-major upload");
        cuda_check(cudaMemcpy(control_device.get(),initial.data(),control_device.bytes(),cudaMemcpyHostToDevice),
            "draft dense control guards");
        cuda_check(cudaMemcpy(candidate_device.get(),initial.data(),candidate_device.bytes(),cudaMemcpyHostToDevice),
            "draft dense candidate guards");
        auto* control=static_cast<std::uint16_t*>(control_device.get())+guard;
        auto* candidate=static_cast<std::uint16_t*>(candidate_device.get())+guard;
        ninfer::exl3::dflash2_dense_t_for_test(
            static_cast<const std::uint16_t*>(input_device.get()),
            static_cast<const std::uint16_t*>(output_major_device.get()),control,
            shape.rows,shape.k,shape.n,false);
        ninfer::exl3::dflash2_dense_t_for_test(
            static_cast<const std::uint16_t*>(input_device.get()),
            static_cast<const std::uint16_t*>(kmajor_device.get()),candidate,
            shape.rows,shape.k,shape.n,true);
        std::vector<std::uint16_t> got_control(initial.size()),got_candidate(initial.size());
        cuda_check(cudaMemcpy(got_control.data(),control_device.get(),control_device.bytes(),cudaMemcpyDeviceToHost),
            "draft dense control download");
        cuda_check(cudaMemcpy(got_candidate.data(),candidate_device.get(),candidate_device.bytes(),cudaMemcpyDeviceToHost),
            "draft dense candidate download");
        require(got_control==got_candidate,"draft dense K-major exact differential");
        for(std::size_t i=0;i<initial.size();++i)
            if(i<guard || i>=guard+output_count)
                require(got_control[i]==canary,"draft dense K-major output guard");
        values+=output_count;++cases;
    }
    std::cout<<"DFLASH2_DENSE_KMAJOR PASS cases="<<cases
        <<" values="<<values
        <<" bit_exact=1 guards=1 physical_geometry=1\n";
}
