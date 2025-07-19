/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
 * Copyright (c) 2003 The Regents of The University of Michigan
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file
 * Definitions of functional page table.
 */
#include "mem/page_table.hh"

#include <string>

#include "base/compiler.hh"
#include "base/trace.hh"
#include "debug/MMU.hh"
#include "sim/faults.hh"
#include "sim/serialize.hh"
#include "debug/NVMainMyDebug.hh"

namespace gem5
{

static const int IMAGE_H          = 32;   // 圖片高度
static const int IMAGE_W          = 32;   // 圖片寬度
static const int IMAGE_C          = 3;     // 圖片通道數
static const int PATCH_SIZE       = 32;    // Patch 大小
static const int D_MODEL          = 384;   // 嵌入維度
static const int D_FF             = 16;    // Feed Forward 隱藏維度
int patchesY         = IMAGE_H / PATCH_SIZE;
int patchesX         = IMAGE_W / PATCH_SIZE;
int numPatches       = patchesY * patchesX;          // 例如 7 * 7 = 49
int patchFlattenDim  = PATCH_SIZE * PATCH_SIZE * IMAGE_C;  // 例如 32*32*3 = 3072
Addr image_data     = 0x10b000;
Addr W_patch_quant  = 0x3fedf000;
Addr b_patch_quant  = 0xa1e000;
Addr CLS_quant      = 0xa1f000;
Addr posEmb_quant   = 0xa1f000;
Addr Wq_quant       = 0x3feba000;
Addr Wk_quant       = 0x3fe95000;
Addr Wv_quant       = 0x3fe70000;
Addr Wo_quant       = 0x3fe4b000;
Addr bias_quant     = 0xec6000;
Addr W1_quant       = 0xec8000;
Addr W2_quant       = 0xec9000;
Addr bias1_2_quant  = 0xa1f000;
Addr bias1_2_quant_2  = 0xecb000;

void
EmulationPageTable::map(Addr vaddr, Addr paddr, int64_t size, uint64_t flags)
{
    // /*
    if(vaddr >= image_data && vaddr <= image_data + IMAGE_H * IMAGE_W * IMAGE_C){
        // DPRINTF(NVMainMyDebug, "image_data %#x in, flag = %llu\n", vaddr, flags);
        flags = flags | EmulationPageTable::Uncacheable;
    }
    if(vaddr >= W_patch_quant && vaddr <= W_patch_quant + patchFlattenDim * D_MODEL){
        // DPRINTF(NVMainMyDebug, "W_patch_quant %#x in, flag = %llu\n", vaddr, flags);
        flags = flags | EmulationPageTable::Uncacheable;
    }
    if(vaddr == b_patch_quant){
        // DPRINTF(NVMainMyDebug, "b_patch_quant %#x in, flag = %llu\n", vaddr, flags);
        flags = flags | EmulationPageTable::Uncacheable;
    }
    if(vaddr == CLS_quant){
        // DPRINTF(NVMainMyDebug, "CLS_quant     %#x in, flag = %llu\n", vaddr, flags);
        flags = flags | EmulationPageTable::Uncacheable;    
    }
    if(vaddr >= posEmb_quant && vaddr <= posEmb_quant + (numPatches + 1) * D_MODEL){
        // DPRINTF(NVMainMyDebug, "posEmb_quant  %#x in, flag = %llu\n", vaddr, flags);
        flags = flags | EmulationPageTable::Uncacheable;    
    }
    if(vaddr >= Wq_quant && vaddr <= Wq_quant + D_MODEL * D_MODEL){
        // DPRINTF(NVMainMyDebug, "Wq_quant      %#x in, flag = %llu\n", vaddr, flags);
        flags = flags | EmulationPageTable::Uncacheable;    
    }
    if(vaddr >= Wk_quant && vaddr <= Wk_quant + D_MODEL * D_MODEL){
        // DPRINTF(NVMainMyDebug, "Wk_quant      %#x in, flag = %llu\n", vaddr, flags);
        flags = flags | EmulationPageTable::Uncacheable;    
    }
    if(vaddr >= Wv_quant && vaddr <= Wv_quant + D_MODEL * D_MODEL){
        // DPRINTF(NVMainMyDebug, "Wv_quant      %#x in, flag = %llu\n", vaddr, flags);
        flags = flags | EmulationPageTable::Uncacheable;    
    }
    if(vaddr >= Wo_quant && vaddr <= Wo_quant + D_MODEL * D_MODEL){
        // DPRINTF(NVMainMyDebug, "Wo_quant      %#x in, flag = %llu\n", vaddr, flags);
        flags = flags | EmulationPageTable::Uncacheable;    
    }
    if(vaddr >= bias_quant && vaddr <= bias_quant + 0x1000){
        // DPRINTF(NVMainMyDebug, "bias_quant    %#x in, flag = %llu\n", vaddr, flags);
        flags = flags | EmulationPageTable::Uncacheable;    
    }
    if(vaddr >= W1_quant && vaddr <= W1_quant + D_MODEL * D_FF){
        // DPRINTF(NVMainMyDebug, "W1_quant      %#x in, flag = %llu\n", vaddr, flags);
        flags = flags | EmulationPageTable::Uncacheable;    
    }
    if(vaddr >= W2_quant && vaddr <= W2_quant + D_MODEL * D_FF){
        // DPRINTF(NVMainMyDebug, "W2_quant      %#x in, flag = %llu\n", vaddr, flags);
        flags = flags | EmulationPageTable::Uncacheable;    
    }
    if(vaddr == bias1_2_quant){
        // DPRINTF(NVMainMyDebug, "bias1_2_quant %#x in, flag = %llu\n", vaddr, flags);
        flags = flags | EmulationPageTable::Uncacheable;    
    }
    if(vaddr == bias1_2_quant_2){
        // DPRINTF(NVMainMyDebug, "bias1_2_quant %#x in, flag = %llu\n", vaddr, flags);
        flags = flags | EmulationPageTable::Uncacheable;    
    }
    // */

    bool clobber = flags & Clobber;
    // starting address must be page aligned
    assert(pageOffset(vaddr) == 0);

    DPRINTF(MMU, "Allocating Page: %#x-%#x\n", vaddr, vaddr + size);

    while (size > 0) {
        auto it = pTable.find(vaddr);
        if (it != pTable.end()) {
            // already mapped
            panic_if(!clobber,
                     "EmulationPageTable::allocate: addr %#x already mapped",
                     vaddr);
            it->second = Entry(paddr, flags);
        } else {
            pTable.emplace(vaddr, Entry(paddr, flags));
        }

        size -= _pageSize;
        vaddr += _pageSize;
        paddr += _pageSize;
    }
}

void
EmulationPageTable::remap(Addr vaddr, int64_t size, Addr new_vaddr)
{
    assert(pageOffset(vaddr) == 0);
    assert(pageOffset(new_vaddr) == 0);

    DPRINTF(MMU, "moving pages from vaddr %08p to %08p, size = %d\n", vaddr,
            new_vaddr, size);

    while (size > 0) {
        [[maybe_unused]] auto new_it = pTable.find(new_vaddr);
        auto old_it = pTable.find(vaddr);
        assert(old_it != pTable.end() && new_it == pTable.end());

        pTable.emplace(new_vaddr, old_it->second);
        pTable.erase(old_it);
        size -= _pageSize;
        vaddr += _pageSize;
        new_vaddr += _pageSize;
    }
}

void
EmulationPageTable::getMappings(std::vector<std::pair<Addr, Addr>> *addr_maps)
{
    for (auto &iter : pTable)
        addr_maps->push_back(std::make_pair(iter.first, iter.second.paddr));
}

void
EmulationPageTable::unmap(Addr vaddr, int64_t size)
{
    assert(pageOffset(vaddr) == 0);

    DPRINTF(MMU, "Unmapping page: %#x-%#x\n", vaddr, vaddr + size);

    while (size > 0) {
        auto it = pTable.find(vaddr);
        assert(it != pTable.end());
        pTable.erase(it);
        size -= _pageSize;
        vaddr += _pageSize;
    }
}

bool
EmulationPageTable::isUnmapped(Addr vaddr, int64_t size)
{
    // starting address must be page aligned
    assert(pageOffset(vaddr) == 0);

    for (int64_t offset = 0; offset < size; offset += _pageSize)
        if (pTable.find(vaddr + offset) != pTable.end())
            return false;

    return true;
}

const EmulationPageTable::Entry *
EmulationPageTable::lookup(Addr vaddr)
{
    Addr page_addr = pageAlign(vaddr);
    PTableItr iter = pTable.find(page_addr);
    if (iter == pTable.end())
        return nullptr;
    return &(iter->second);
}

bool
EmulationPageTable::translate(Addr vaddr, Addr &paddr)
{
    const Entry *entry = lookup(vaddr);
    if (!entry) {
        DPRINTF(MMU, "Couldn't Translate: %#x\n", vaddr);
        return false;
    }
    paddr = pageOffset(vaddr) + entry->paddr;
    DPRINTF(MMU, "Translating: %#x->%#x\n", vaddr, paddr);
    return true;
}

Fault
EmulationPageTable::translate(const RequestPtr &req)
{
    Addr paddr;
    assert(pageAlign(req->getVaddr() + req->getSize() - 1) ==
           pageAlign(req->getVaddr()));
    if (!translate(req->getVaddr(), paddr))
        return Fault(new GenericPageTableFault(req->getVaddr()));
    req->setPaddr(paddr);
    if ((paddr & (_pageSize - 1)) + req->getSize() > _pageSize) {
        panic("Request spans page boundaries!\n");
        return NoFault;
    }
    return NoFault;
}

void
EmulationPageTable::PageTableTranslationGen::translate(Range &range) const
{
    const Addr page_size = pt->pageSize();

    Addr next = roundUp(range.vaddr, page_size);
    if (next == range.vaddr)
        next += page_size;
    range.size = std::min(range.size, next - range.vaddr);

    if (!pt->translate(range.vaddr, range.paddr))
        range.fault = Fault(new GenericPageTableFault(range.vaddr));
}

void
EmulationPageTable::serialize(CheckpointOut &cp) const
{
    ScopedCheckpointSection sec(cp, "ptable");
    paramOut(cp, "size", pTable.size());

    PTable::size_type count = 0;
    for (auto &pte : pTable) {
        ScopedCheckpointSection sec(cp, csprintf("Entry%d", count++));

        paramOut(cp, "vaddr", pte.first);
        paramOut(cp, "paddr", pte.second.paddr);
        paramOut(cp, "flags", pte.second.flags);
    }
    assert(count == pTable.size());
}

void
EmulationPageTable::unserialize(CheckpointIn &cp)
{
    int count;
    ScopedCheckpointSection sec(cp, "ptable");
    paramIn(cp, "size", count);

    for (int i = 0; i < count; ++i) {
        ScopedCheckpointSection sec(cp, csprintf("Entry%d", i));

        Addr vaddr;
        UNSERIALIZE_SCALAR(vaddr);
        Addr paddr;
        uint64_t flags;
        UNSERIALIZE_SCALAR(paddr);
        UNSERIALIZE_SCALAR(flags);

        pTable.emplace(vaddr, Entry(paddr, flags));
    }
}

const std::string
EmulationPageTable::externalize() const
{
    std::stringstream ss;
    for (PTable::const_iterator it=pTable.begin(); it != pTable.end(); ++it) {
        ss << std::hex << it->first << ":" << it->second.paddr << ";";
    }
    return ss.str();
}

} // namespace gem5
