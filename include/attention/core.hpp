#pragma once

namespace attention {

// Returns zero and exists solely so downstream code can link against
// attention_core while the correctness-first engine is scaffolded.
int placeholder();

}  // namespace attention
