#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int
load_images(const char *fname, int count, unsigned char *labels, unsigned char (*images)[784])
{
    char line_buf[1024*4];
    char *token;
    FILE *f;
    int i, j;
    f = fopen(fname, "r");
    for (i = 0; i < count; i++) {
        fgets(line_buf, sizeof(line_buf), f);
        token = strtok(line_buf, ",");
        labels[i] = atoi(token);
        for (j = 0; j < 784; j++) {
            token = strtok(NULL, ",");
            images[i][j] = atoi(token);
        }
    }
    fclose(f);
    return 0;
}

void
print_image(const unsigned char image[784])
{
    const char *glyphs = " .:-=+*#%@";
    unsigned char v;
    int x, y;
    for (x = 0; x < 28; x++) {
        for (y = 0; y < 28; y++) {
            v = image[x*28+y];
            putchar(glyphs[v * strlen(glyphs) / 256]);
        }
        putchar('\n');
    }
}
