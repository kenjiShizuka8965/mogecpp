#include <moge_ggml/moge_c.h>
#include <stdio.h>

int main(void) {
    moge_load_options_t l = moge_default_load_options();
    moge_infer_options_t i = moge_default_infer_options();
    if (l.backend != MOGE_BACKEND_AUTO || l.cpu_fallback != 1 || l.op_offload != 1) return 2;
    if (i.resolution_level != 9 || i.refine_steps != 3 || i.apply_mask != 1) return 3;
    if (MOGE_API_VERSION != 1) return 4;
    if (MOGE_VERSION_MAJOR != 0 || MOGE_VERSION_MINOR != 4 || MOGE_VERSION_PATCH != 2) return 5;
    if (!moge_version_string() || moge_version_string()[0] != '0') return 6;
    puts("moge_c_api=ok");
    return 0;
}
