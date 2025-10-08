run:
	$(MAKE) -C gpu run

cont:
	$(MAKE) -C gpu cont

clean:
	rm -f *.out *.png
	$(MAKE) -C gpu clean